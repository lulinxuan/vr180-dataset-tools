#!/usr/bin/env python3
"""Rate-quality curves of all 24 clips and the per-region BD-rate heat map from the study summary."""
import argparse
import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR','/tmp/vr180-plot-mpl')
os.environ.setdefault('XDG_CACHE_HOME','/tmp/vr180-plot-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator,FuncFormatter,NullFormatter
import numpy as np

REGIONS=['all','center','lateral','upper','lower']

def render(data,out,title):
    cases=data['scenes'];n=len(cases);cols=min(4,n);rows=math.ceil(n/cols)
    fig,axes=plt.subplots(rows,cols,figsize=(cols*3.45,rows*2.85+.65),squeeze=False,layout='constrained')
    for index,(ax,case) in enumerate(zip(axes.flat,cases),1):
        label=f'S{index:02d}'
        for rep,color in [('fisheye','#0072B2'),('eq','#D55E00')]:
            points=sorted([p for p in case['points'] if p['representation']==rep],key=lambda p:p['stereo_mbps'])
            ax.plot([p['stereo_mbps'] for p in points],[p['regions']['all']['psnr_y_db'] for p in points],'-o',markersize=3.5,color=color,label=rep)
        ax.set_title(f'{label} · {case["scene_type_label"]}',fontsize=9)
        ax.set_xscale('log');ax.set_xlabel('Stereo video payload rate (Mb/s)',fontsize=8)
        ax.set_ylabel("Common-viewport Y′ PSNR (dB)",fontsize=8);ax.grid(alpha=.2);ax.tick_params(labelsize=8)
        ax.xaxis.set_major_locator(LogLocator(base=10,subs=[1,2,5]));ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:g}'))
        ax.xaxis.set_minor_formatter(NullFormatter());ax.legend(fontsize=8)
    for ax in list(axes.flat)[n:]:ax.set_visible(False)
    fig.suptitle(title,fontsize=11)
    fig.savefig(out/'rate-quality-all-cases.png',dpi=180);fig.savefig(out/'rate-quality-all-cases.pdf');plt.close(fig)
    matrix=np.full((n,len(REGIONS)),np.nan)
    table=[]
    for i,case in enumerate(cases):
        row={'case':f'S{i+1:02d}','sample_id':case['sample_id'],'scene_type_label':case['scene_type_label']}
        for j,region in enumerate(REGIONS):
            value=case['rate_differences'][region]
            if value['status']=='available':matrix[i,j]=value['fisheye_vs_eq_percent']
            row[region+'_percent']=float(matrix[i,j]) if np.isfinite(matrix[i,j]) else ''
            row[region+'_reason']='' if value['status']=='available' else value['reason']
        table.append(row)
    magnitude=max(1,float(np.nanmax(np.abs(matrix)))) if np.isfinite(matrix).any() else 1
    fig,ax=plt.subplots(figsize=(8,max(3,n*.32+1.9)),layout='constrained')
    cmap=plt.get_cmap('coolwarm').copy();cmap.set_bad('#dddddd')
    im=ax.imshow(matrix,cmap=cmap,vmin=-magnitude,vmax=magnitude,aspect='auto')
    ax.set_xticks(range(5),['All','Center','Lateral','Upper','Lower'])
    ax.set_yticks(range(n),[f'S{i+1:02d}  {s["scene_type_label"]}' for i,s in enumerate(cases)],fontsize=8)
    for i in range(n):
        for j in range(5):ax.text(j,i,f'{matrix[i,j]:+.1f}' if np.isfinite(matrix[i,j]) else 'N/A',ha='center',va='center',fontsize=8,
            color='white' if np.isfinite(matrix[i,j]) and abs(matrix[i,j])>magnitude*.7 else 'black')
    ax.set_title(title+'\nMatched-quality rate difference (%)',fontsize=10)
    fig.colorbar(im,ax=ax,label='Rate difference (%)')
    fig.supxlabel('Negative means less fisheye rate. Each region uses its own common PSNR interval.\nN/A values are retained, not zero.',fontsize=8)
    fig.savefig(out/'rate-difference-by-region.png',dpi=180);fig.savefig(out/'rate-difference-by-region.pdf');plt.close(fig)
    with (out/'rate-differences.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--summary',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    data=json.loads(args.summary.read_text())
    if not data['all_selected_samples_complete'] or data['selected_samples']!=24 or len(data['scenes'])!=24:
        ap.error('The summary does not contain all 24 selected clips')
    if any(s['status']!='complete' for s in data['scenes']):ap.error('A selected clip is not complete')
    args.out.mkdir(parents=True,exist_ok=True)
    render(data,args.out,'24 stereo clips, 3 s per eye, 9 viewports')
    print(json.dumps({'case_count':len(data['scenes'])}))

if __name__=='__main__':main()
