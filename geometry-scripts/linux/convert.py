#!/usr/bin/env python3
"""Calibrated native fisheye <-> half-EQ using portable lookup tables.

Image inputs: PNG/TIFF/JPEG. Video inputs: already decoded/rendered eye files,
not BRAW. Defaults to lossless FFV1 for video; explicit HEVC is also available.
"""
from pathlib import Path
import argparse
from fractions import Fraction
import json
import shutil
import subprocess
import tempfile
import time
import cv2
import numpy as np
from geometry import LensMap, file_hash, calibration_object, canonical
import hashlib

INTERPOLATION={'linear':cv2.INTER_LINEAR,'lanczos4':cv2.INTER_LANCZOS4}


def size(text):
    try:
        values=tuple(map(int,text.lower().split('x')))
        if len(values)!=2 or min(values)<2 or max(values)>=32767:raise ValueError()
        return values
    except ValueError:
        raise argparse.ArgumentTypeError('Expected WIDTHxHEIGHT, each 2..32766')


def probe(path,ffprobe):
    return json.loads(subprocess.check_output([ffprobe,'-v','error','-show_streams',
        '-show_format','-of','json',str(path)]))


def video_convert(args,coordinates,output_size,source_probe,partial):
    ffmpeg=shutil.which(args.ffmpeg)
    if not ffmpeg:raise ValueError('FFmpeg is not on PATH')
    v=next(s for s in source_probe['streams'] if s['codec_type']=='video')
    if Fraction(v['r_frame_rate'])!=Fraction(v['avg_frame_rate']):
        raise ValueError('This reader supports CFR only; variable-rate inputs need timestamp-aware processing')
    fps=v['avg_frame_rate'];iw,ih=v['width'],v['height'];ow,oh=output_size
    if v.get('pix_fmt','').startswith('gbrp'):
        decode_filter='format=bgr48le'
    elif v.get('color_space')=='bt709' and v.get('color_range')=='tv':
        decode_filter='scale=in_color_matrix=bt709:in_range=tv:out_range=pc,format=bgr48le'
    else:
        raise ValueError('Video path supports dataset BT.709 limited-range YUV or full-range planar RGB intermediates')
    decoder=[ffmpeg,'-v','error','-nostdin','-i',str(args.input),'-map','0:v:0','-an','-sn','-dn',
        '-fps_mode','passthrough','-vf',decode_filter]
    # Stop the producer itself at the requested count. Terminating FFmpeg while
    # its rawvideo stdout is full can block shutdown indefinitely.
    if args.frames is not None:decoder+=['-frames:v',str(args.frames)]
    decoder+=['-f','rawvideo','-pix_fmt','bgr48le','pipe:1']
    encoder=[ffmpeg,'-v','error','-nostdin','-n','-f','rawvideo','-pix_fmt','bgr48le',
        '-s',f'{ow}x{oh}','-r',fps,'-i','pipe:0']
    copy_audio=args.frames is None
    if copy_audio:encoder+=['-i',str(args.input),'-map','0:v:0','-map','1:a?','-c:a','copy']
    else:encoder+=['-map','0:v:0','-an']
    if args.codec=='ffv1':
        encoder+=['-c:v','ffv1','-level','3','-pix_fmt','gbrp16le','-color_range','pc']
    else:
        if ow%2 or oh%2:raise ValueError('HEVC 4:2:0 needs even output dimensions')
        encoder+=['-vf','scale=in_range=pc:out_range=tv:out_color_matrix=bt709,format=yuv420p10le',
            '-c:v',args.codec,'-profile:v','main10','-color_primaries','bt709','-colorspace','bt709','-color_range','tv']
        if args.codec=='libx265':encoder+=['-preset','medium','-crf',str(args.crf)]
        else:encoder+=['-preset','p5','-b:v',args.bitrate]
        if partial.suffix.lower() in ('.mov','.mp4'):encoder+=['-tag:v','hvc1','-movflags','+faststart']
    encoder += ['-fps_mode','passthrough',str(partial)]
    frame_bytes=iw*ih*3*2
    count=0
    with tempfile.TemporaryFile() as de, tempfile.TemporaryFile() as en:
        dec=subprocess.Popen(decoder,stdout=subprocess.PIPE,stderr=de)
        enc=subprocess.Popen(encoder,stdin=subprocess.PIPE,stderr=en)
        try:
            while args.frames is None or count<args.frames:
                chunks=[];remaining=frame_bytes
                while remaining:
                    chunk=dec.stdout.read(remaining)
                    if not chunk:break
                    chunks.append(chunk);remaining-=len(chunk)
                if not chunks:break
                if remaining:raise RuntimeError('Decoder returned an incomplete frame')
                image=np.frombuffer(b''.join(chunks),'<u2').reshape(ih,iw,3)
                warped=cv2.remap(image,coordinates,None,INTERPOLATION[args.interpolation],borderMode=cv2.BORDER_CONSTANT)
                enc.stdin.write(warped.astype('<u2',copy=False).tobytes())
                count+=1
            enc.stdin.close()
            dec.stdout.close()
            dc=dec.wait();ec=enc.wait()
            if ec or dc:
                de.seek(0);en.seek(0)
                raise RuntimeError('FFmpeg failed: '+de.read().decode(errors='replace')+en.read().decode(errors='replace'))
            expected=int(v['nb_frames']) if v.get('nb_frames','').isdigit() else None
            if count==0 or (expected is not None and args.frames is None and count!=expected):
                raise ValueError(f'Frame-count mismatch: processed {count}, expected {expected}')
            return {'frames':count,'fps':fps,'audio_copied':copy_audio,
                'pixel_processing':'Input -> full 16-bit RGB -> geometric remap; BT.709 YUV matrix/range conversion when input is YUV; no transfer-function transform; HEVC output converts to 10-bit YUV',
                'decoder':decoder,'encoder':encoder}
        finally:
            for proc in (dec,enc):
                if proc.poll() is None:proc.kill();proc.wait()
            dec.stdout.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--map',type=Path,help='Defaults to bundled map matching the full calibration fingerprint')
    p.add_argument('--calibration',required=True,type=Path,help='ILPD JSON or dataset metadata.json')
    p.add_argument('--eye',required=True,choices=['left','right'])
    p.add_argument('--direction',required=True,choices=['fisheye-to-eq','eq-to-fisheye'])
    p.add_argument('--input',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--output-size',type=size)
    p.add_argument('--allow-resized-native',action='store_true',help='Assert input fisheye is a full-frame resize without crop/rotation')
    p.add_argument('--allow-nonsquare-eq',action='store_true',help='Assert the entire rectangular raster covers the same half-EQ coordinate domain')
    p.add_argument('--interpolation',choices=list(INTERPOLATION),default='linear',help='Resampling kernel; lanczos4 keeps more fine detail but is slower')
    p.add_argument('--codec',choices=['ffv1','libx265','hevc_nvenc'],default='ffv1')
    p.add_argument('--frames',type=int,help='Convert only the first N frames (audio is not copied)')
    p.add_argument('--crf',type=int,default=18);p.add_argument('--bitrate',default='400M')
    p.add_argument('--ffmpeg',default='ffmpeg');p.add_argument('--ffprobe',default='ffprobe')
    args=p.parse_args()
    if args.frames is not None and args.frames<1:p.error('--frames must be positive')
    output=args.output;mask_path=output.with_name(output.name+'.valid.png');record_path=output.with_name(output.name+'.json')
    partial=output.with_name(output.stem+'.partial'+output.suffix)
    if any(x.exists() for x in (output,mask_path,record_path,partial)):
        p.error('Output or its sidecars already exist; choose a new destination')
    if args.map is None:
        digest=hashlib.sha256(canonical(calibration_object(args.calibration))).hexdigest()
        args.map=Path(__file__).resolve().parent/'maps'/(digest+'.json')
        if not args.map.is_file():
            raise ValueError('No bundled map matches this ILPD. Supply a matching --map; this runtime never invokes an Apple framework')
    lens=LensMap(args.map,args.eye,args.calibration)
    image_mode=args.input.suffix.lower() in {'.png','.tif','.tiff','.jpg','.jpeg'}
    source_probe=None
    if image_mode:
        image=cv2.imread(str(args.input),cv2.IMREAD_UNCHANGED)
        if image is None:raise ValueError('Cannot read image')
        input_size=image.shape[1],image.shape[0]
        if output.suffix.lower() not in ('.png','.tif','.tiff'):
            raise ValueError('Use PNG/TIFF output to avoid an implicit 8-bit JPEG conversion')
    else:
        fp=shutil.which(args.ffprobe)
        if not fp:raise ValueError('ffprobe is not on PATH')
        source_probe=probe(args.input,fp)
        video=next(x for x in source_probe['streams'] if x['codec_type']=='video')
        input_size=video['width'],video['height']
        if args.codec=='ffv1' and output.suffix.lower()!='.mkv':
            raise ValueError('FFV1 output must use .mkv')
        if args.codec!='ffv1' and output.suffix.lower() not in ('.mkv','.mov','.mp4'):
            raise ValueError('HEVC output must use .mkv, .mov or .mp4')
    if args.direction=='fisheye-to-eq':
        if input_size!=lens.source_size and not args.allow_resized_native:
            raise ValueError('Native raster differs from calibration. Explicitly assert full-frame resize with --allow-resized-native, or supply correct native pixels')
        output_size=args.output_size or (7200,7200)
        if output_size[0]!=output_size[1]:raise ValueError('This tool outputs square half-EQ')
    else:
        if input_size[0]!=input_size[1] and not args.allow_nonsquare_eq:
            raise ValueError('Rectangular EQ needs an explicit full-domain assertion --allow-nonsquare-eq')
        output_size=args.output_size or lens.source_size
        if output_size[0]*lens.source_size[1]!=output_size[1]*lens.source_size[0]:
            raise ValueError('Native output must retain calibrated aspect ratio')
    started=time.monotonic()
    coordinates,valid=lens.remap_coordinates(args.direction,input_size,output_size)
    map_seconds=time.monotonic()-started
    output.parent.mkdir(parents=True,exist_ok=True)
    if image_mode:
        warped=cv2.remap(image,coordinates,None,INTERPOLATION[args.interpolation],borderMode=cv2.BORDER_CONSTANT)
        if not cv2.imwrite(str(partial),warped):raise ValueError('Image write failed')
        result={'frames':1,'dtype':str(warped.dtype)}
    else:result=video_convert(args,coordinates,output_size,source_probe,partial)
    partial.rename(output)
    if not cv2.imwrite(str(mask_path),valid.astype(np.uint8)*255):raise ValueError('Mask write failed')
    result.update({'schema':'vr180.conversion.v1','direction':args.direction,'eye':args.eye,
        'interpolation':args.interpolation,
        'calibration_sha256':lens.digest,'map_sha256':lens.info['eyes'][args.eye]['sha256'],
        'input_size':input_size,'output_size':output_size,'map_grid_size':[lens.width,lens.height],
        'valid_fraction':float(valid.mean()),'coordinate_map_seconds':map_seconds,
        'elapsed_seconds':time.monotonic()-started,'output_sha256':file_hash(output),
        'projection':'half_equirectangular' if args.direction=='fisheye-to-eq' else 'native_fisheye',
        'projection_signaling':'This JSON is authoritative; generic output containers are not automatically tagged for VR playback',
        'limitations':'Static per-sample calibration; no BRAW decode; no semantic mask/Apple feathering; no reconstruction outside EQ coverage; not bitwise reversible or an exact match to Resolve grading/projection'})
    record_path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
