"""Portable calibration-table geometry. No Apple/Metal/Swift runtime imports.

Tables map half-EQ output pixel centers to normalized native lens coordinates.
Inverse maps recover only the represented half-EQ domain, never unseen pixels.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.spatial import cKDTree


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode('utf-8')


def calibration_object(path):
    value = json.loads(Path(path).read_text())
    if 'calibration' in value:
        value = value['calibration']['ilpd']
    if isinstance(value, str):
        value = json.loads(value)
    return value


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(8*1024**2), b''):
            h.update(data)
    return h.hexdigest()


class LensMap:
    def __init__(self, manifest, eye, calibration):
        self.path = Path(manifest).resolve()
        self.info = json.loads(self.path.read_text())
        if self.info.get('schema') != 'vr180.stmap.v1':
            raise ValueError('Unsupported map schema')
        if eye not in ('left', 'right'):
            raise ValueError('Eye must be left or right')
        cal = calibration_object(calibration)
        digest = hashlib.sha256(canonical(cal)).hexdigest()
        if digest != self.info['calibration_sha256']:
            raise ValueError('Full ILPD fingerprint does not match this map')
        self.eye, self.digest = eye, digest
        record = self.info['eyes'][eye]
        path = (self.path.parent/record['file']).resolve()
        if not path.is_relative_to(self.path.parent):
            raise ValueError('Map path escapes package directory')
        if path.stat().st_size != record['bytes'] or file_hash(path) != record['sha256']:
            raise ValueError('Map integrity check failed')
        self.uv = np.load(path, mmap_mode='r', allow_pickle=False)
        self.width, self.height = self.info['grid_size']
        if self.uv.shape != (self.height, self.width, 2) or self.uv.dtype != np.dtype('<f4'):
            raise ValueError('Map must be HxWx2 little-endian float32')
        if min(self.width, self.height) < 2 or not np.isfinite(self.uv).all():
            raise ValueError('Map grid is too small or contains nonfinite coordinates')
        self.source_size = tuple(self.info['source_sizes'][eye])
        view = next(v for v in cal['captureDevice']['views'] if v['viewDescription']==eye)
        if tuple(view['imageSize']) != self.source_size:
            raise ValueError('Map source dimensions differ from calibration')
        self._tree = None

    def sample(self, u, v, derivatives=False):
        """Float64 bilinear lookup; domain excludes extrapolation beyond grid centers."""
        u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
        valid = np.isfinite(u) & np.isfinite(v) & (u>=0) & (v>=0) & (u<=self.width-1) & (v<=self.height-1)
        uc = np.clip(np.nan_to_num(u), 0, self.width-1)
        vc = np.clip(np.nan_to_num(v), 0, self.height-1)
        ix = np.minimum(uc.astype(np.int64), self.width-2)
        iy = np.minimum(vc.astype(np.int64), self.height-2)
        tx, ty = (uc-ix)[...,None], (vc-iy)[...,None]
        a = np.asarray(self.uv[iy,ix], float)
        b = np.asarray(self.uv[iy,ix+1], float)
        c = np.asarray(self.uv[iy+1,ix], float)
        d = np.asarray(self.uv[iy+1,ix+1], float)
        value = (1-ty)*((1-tx)*a+tx*b)+ty*((1-tx)*c+tx*d)
        if derivatives:
            return value, (1-ty)*(b-a)+ty*(d-c), (1-tx)*(c-a)+tx*(d-b), valid
        return value, valid

    def _inverse_seed(self):
        if self._tree is not None:
            return
        xs = np.unique(np.append(np.arange(0,self.width,16),self.width-1))
        ys = np.unique(np.append(np.arange(0,self.height,16),self.height-1))
        xx, yy = np.meshgrid(xs,ys)
        values = np.asarray(self.uv[yy,xx]).reshape(-1,2)*self.source_size-.5
        self._seed_uv = np.stack([xx,yy],-1).reshape(-1,2).astype(float)
        self._tree = cKDTree(values)

    def inverse_points(self, native_pixels, tolerance=.25, iterations=20):
        """Return grid coordinates, validity and source-pixel residual per point."""
        self._inverse_seed()
        points = np.asarray(native_pixels,float)
        shape = points.shape[:-1]
        points = points.reshape(-1,2)
        finite = np.isfinite(points).all(axis=1)
        safe = np.where(finite[:,None],points,0)
        _, index = self._tree.query(safe, workers=1)
        uv = self._seed_uv[index].copy()
        target = (safe+.5)/self.source_size
        nonsingular = np.ones(len(points),bool)
        for _ in range(iterations):
            value, dx, dy, _ = self.sample(uv[:,0],uv[:,1],True)
            error = value-target
            det = dx[:,0]*dy[:,1]-dy[:,0]*dx[:,1]
            good = abs(det)>1e-15
            nonsingular = good
            denom = np.where(good,det,1)
            du = (error[:,0]*dy[:,1]-error[:,1]*dy[:,0])/denom
            dv = (dx[:,0]*error[:,1]-dx[:,1]*error[:,0])/denom
            uv[:,0] = np.clip(uv[:,0]-np.where(good,np.clip(du,-64,64),0),0,self.width-1)
            uv[:,1] = np.clip(uv[:,1]-np.where(good,np.clip(dv,-64,64),0),0,self.height-1)
            if np.max(np.abs(np.stack([du,dv],-1))[good],initial=0)<1e-7:
                break
        value, in_map = self.sample(uv[:,0],uv[:,1])
        residual = np.linalg.norm((value-target)*self.source_size,axis=1)
        native_bounds = (safe[:,0]>=0)&(safe[:,1]>=0)&(safe[:,0]<=self.source_size[0]-1)&(safe[:,1]<=self.source_size[1]-1)
        valid = finite & in_map & nonsingular & native_bounds & (residual<=tolerance)
        return uv.reshape(*shape,2),valid.reshape(shape),residual.reshape(shape)

    def remap_coordinates(self, direction, input_size, output_size, block_rows=64):
        """OpenCV inverse-warp arrays; input/output sizes are(width,height)."""
        iw,ih = map(int,input_size);ow,oh = map(int,output_size)
        if min(iw,ih,ow,oh)<2 or max(iw,ih,ow,oh)>=32767:
            raise ValueError('Dimensions must be2..32766 for OpenCV remap')
        if direction not in ('fisheye-to-eq','eq-to-fisheye'):
            raise ValueError('Unknown conversion direction')
        result = np.empty((oh,ow,2),np.float32)
        mask = np.empty((oh,ow),bool)
        for start in range(0,oh,block_rows):
            yy,xx = np.indices((min(block_rows,oh-start),ow),dtype=float)
            yy += start
            if direction=='fisheye-to-eq':
                uv,ok = self.sample((xx+.5)*self.width/ow-.5,(yy+.5)*self.height/oh-.5)
                xy = uv*np.array([iw,ih])-.5
            else:
                native = np.stack([(xx+.5)*self.source_size[0]/ow-.5,
                                   (yy+.5)*self.source_size[1]/oh-.5],-1)
                uv,ok,error = self.inverse_points(native)
                xy = (uv+.5)/np.array([self.width,self.height])*np.array([iw,ih])-.5
            ok &= (xy[...,0]>=0)&(xy[...,0]<=iw-1)&(xy[...,1]>=0)&(xy[...,1]<=ih-1)
            result[start:start+len(yy)] = np.where(ok[...,None],xy,-100)
            mask[start:start+len(yy)] = ok
        return result,mask
