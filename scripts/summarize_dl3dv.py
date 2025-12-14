import numpy as np 
from glob import glob 
import os 
import fire 
from tabulate import tabulate





def summarize(dir):

    scenes = os.listdir(dir)
    scenes.sort()

    stat_dict= {}
    # res_names = ['ours_chkpnt_latest.pth', 'ours_refine_0_chkpnt10000.pth']
    res_names = ['ours_chkpnt10000.pth', 'ours_refine_0_chkpnt10000.pth', 'ours_refine_1_chkpnt10000.pth']

    for res_name in res_names:
        stat_dict[res_name] = {}
        for scene in scenes:
            stat_dict[res_name][scene] = {}
            res_path = os.path.join(dir, scene, 'eval_res.txt')
            if not os.path.exists(res_path):
                print(f"res_path {res_path} not exist")
                continue
            with open(res_path, 'r') as f:
                lines = f.readlines()
        
            for i, line in enumerate(lines):
            
                # for res_name in res_names:
                if res_name in line:
                    stat_dict[res_name][scene] = {}
                    for sub_line in lines[i+1:i+4]:
                        if "SSIM" in sub_line:
                            ssim = float(sub_line.split(':')[1].strip())

                        if 'PSNR' in sub_line:
                            psnr = float(sub_line.split(':')[1].strip())
                    
                        if 'LPIPS' in sub_line:
                            lpips = float(sub_line.split(':')[1].strip())
                
                    stat_dict[res_name][scene]['SSIM'] = ssim
                    stat_dict[res_name][scene]['PSNR'] = psnr
                    stat_dict[res_name][scene]['LPIPS'] = lpips
                    break # no need to check other res_name
        
    
    # table = [['scene', 'SSIM', 'PSNR', 'LPIPS']]
    table = []
    for res_name in res_names:
        avg_ssim = 0
        avg_psnr = 0
        avg_lpips = 0
        cnt=  0
        table.append([res_name, 'SSIM', 'PSNR', 'LPIPS'])
        for scene in scenes:
            if scene not in stat_dict[res_name]:
                continue
            
            try:
                ssim = stat_dict[res_name][scene]['SSIM']
                psnr = stat_dict[res_name][scene]['PSNR']
                lpips = stat_dict[res_name][scene]['LPIPS']
                avg_ssim += ssim
                avg_psnr += psnr   
                avg_lpips += lpips
                cnt+=1
            except:
                print(f"Error in {res_name} {scene}")
                continue
            table.append([scene, ssim, psnr, lpips])

        table.append([f'AVG({cnt} scenes)', avg_ssim/cnt, avg_psnr/cnt, avg_lpips/cnt])


    print(tabulate(table, headers=[], tablefmt="grid"))

if __name__ == '__main__':
    fire.Fire(summarize)