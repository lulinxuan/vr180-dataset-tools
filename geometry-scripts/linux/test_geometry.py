import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np
from geometry import LensMap,canonical

ROOT=Path(__file__).resolve().parent


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name)
        self.cal={'captureDevice':{'views':[{'viewDescription':'left','imageSize':[64,48]},
                                          {'viewDescription':'right','imageSize':[64,48]}]}}
        self.calpath=self.base/'cal.json';self.calpath.write_bytes(canonical(self.cal))
        y,x=np.indices((48,64),dtype=np.float32)
        uv=np.stack([(x+.5)/64,(y+.5)/48],-1).astype('<f4')
        path=self.base/'left.npy';np.save(path,uv)
        m={'schema':'vr180.stmap.v1','calibration_sha256':hashlib.sha256(canonical(self.cal)).hexdigest(),
           'grid_size':[64,48],'source_sizes':{'left':[64,48]},
           'eyes':{'left':{'file':'left.npy','bytes':path.stat().st_size,
                           'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}}}
        self.manifest=self.base/'map.json';self.manifest.write_text(json.dumps(m))

    def tearDown(self):self.temp.cleanup()

    def lens(self):return LensMap(self.manifest,'left',self.calpath)

    def test_synthetic_identity_and16bit(self):
        lens=self.lens();coords,valid=lens.remap_coordinates('fisheye-to-eq',(64,48),(64,48))
        y,x=np.indices((48,64),dtype=np.float32)
        np.testing.assert_allclose(coords[2:-2,2:-2],np.stack([x,y],-1)[2:-2,2:-2],atol=1e-5)
        image=(np.arange(48*64).reshape(48,64)*20).astype(np.uint16)
        warped=cv2.remap(image,coords,None,cv2.INTER_LINEAR)
        self.assertEqual(warped.dtype,np.uint16)
        np.testing.assert_array_equal(warped[2:-2,2:-2],image[2:-2,2:-2])

    def test_inverse_and_out_of_domain(self):
        points=np.array([[5.125,7.45],[32,24],[1000,-20],[np.nan,3]])
        uv,valid,error=self.lens().inverse_points(points)
        np.testing.assert_allclose(uv[:2],points[:2],atol=1e-5)
        self.assertEqual(valid.tolist(),[True,True,False,False])

    def test_wrong_calibration_is_rejected(self):
        cal=copy.deepcopy(self.cal);cal['different']='calibration'
        self.calpath.write_text(json.dumps(cal))
        with self.assertRaisesRegex(ValueError,'fingerprint'):self.lens()

    def test_corrupt_map_is_rejected(self):
        with (self.base/'left.npy').open('ab') as f:f.write(b'X')
        with self.assertRaisesRegex(ValueError,'integrity'):self.lens()

    def test_wrong_eye_is_not_silently_reused(self):
        with self.assertRaises((KeyError,ValueError)):LensMap(self.manifest,'right',self.calpath)

    def test_real_both_calibrations_both_eyes(self):
        digests=sorted(path.name.split('.')[0] for path in (ROOT/'calibrations').glob('*.ilpd.json'))
        self.assertEqual(len(digests),2)
        rng=np.random.default_rng(719)
        for digest in digests:
            for eye in ['left','right']:
                lens=LensMap(ROOT/'maps'/(digest+'.json'),eye,ROOT/'calibrations'/(digest+'.ilpd.json'))
                uv=rng.uniform([40,40],[lens.width-40,lens.height-40],(2000,2))
                mapped,_=lens.sample(uv[:,0],uv[:,1]);points=mapped*lens.source_size-.5
                inside=((points>=0)&(points<=np.array(lens.source_size)-1)).all(1)
                recovered,valid,error=lens.inverse_points(points[inside])
                self.assertTrue(valid.all(),(eye,float(valid.mean())))
                self.assertLess(float(error.max()),.01)
                # Reject solutions on another fold of a map, not just small forward error.
                np.testing.assert_allclose(recovered,uv[inside],atol=.05)


if __name__=='__main__':unittest.main(verbosity=2)
