import os
import sys
import json
import argparse

from imatools.common import itktools as itku
from imatools.common import config

logger = config.configure_logging(log_name=__name__)

class ScarQuantificationTools :

    def __init__(self, cemrg_folder = "", mirtk_folder = "", scar_cmd_name = 'MitkCemrgScarProjectionOptions', clip_cmd_name = 'MitkCemrgApplyExternalClippers', scar_method='iir') -> None:
        self._cemrg = cemrg_folder
        self._mirtk = mirtk_folder
        self._scar_cmd_name = scar_cmd_name
        self._clip_cmd_name = clip_cmd_name
        self._scar_method = scar_method

    @property
    def cemrg(self):
        return self._cemrg
    
    @property
    def mirtk(self):
        return self._mirtk
    
    @property
    def scar_cmd_name(self):
        return self._scar_cmd_name
    
    @property
    def clip_cmd_name(self):
        return self._clip_cmd_name
    
    @property
    def scar_method(self):
        return self._scar_method
    
    @cemrg.setter
    def cemrg(self, cemrg_folder):
        self._cemrg = cemrg_folder
    
    @mirtk.setter
    def mirtk(self, mirtk_folder):
        self._mirtk = mirtk_folder
    
    @scar_cmd_name.setter
    def scar_cmd_name(self, scar_cmd_name):
        self._scar_cmd_name = scar_cmd_name

    @clip_cmd_name.setter
    def clip_cmd_name(self, clip_cmd_name):
        self._clip_cmd_name = clip_cmd_name
    
    @scar_method.setter
    def scar_method(self, scar_method):
        if scar_method != 'iir' and scar_method != 'msd':
            logger.error("Error: Method must be 'iir' or 'msd'")
            raise ValueError("Error: Method must be 'iir' or 'msd'")
        self._scar_method = scar_method

    def get_scar_method(self):
        midic = {
            'iir' : 1,
            'msd' : 2
        }
        return midic[self._scar_method]

    def check_mirtk(self, test="close-image") -> bool:
        """Check if MIRTK is installed"""
        res = False
        test_cmd = os.path.join(self.mirtk, test)
        if os.path.isfile(test_cmd) or os.path.isfile(f'{test_cmd}.exe') :
            res = True
        
        return res
    
    def check_cemrg(self, test="MitkCemrgScarProjectionOptions") -> bool:
        """Check if CEMRG is installed"""
        res = False
        test_cmd = os.path.join(self.cemrg, test)
        if os.path.isfile(test_cmd):
            res = True
        
        return res    
    
    def run_cmd(self, script_dir, cmd_name, arguments, debug=True):
        """ Return the command to execute"""
        cmd_name = os.path.join(script_dir, cmd_name) if script_dir != '' else cmd_name
        cmd = f'{cmd_name} '
        cmd += ' '.join(arguments)
        stst = 0

        if debug:
            logger.info(cmd)
        else:
            stst = os.system(cmd)

        return stst, cmd
    
    def run_scar(self, arguments, debug=False):
        return self.run_cmd(self._cemrg, self._scar_cmd_name, arguments, debug)
    
    def run_clip(self, arguments, debug=False):
        return self.run_cmd(self._cemrg, self._clip_cmd_name, arguments, debug)
    
    def create_segmentation_mesh(self, dir: str, pveins_file='PVeinsCroppedImage.nii', 
                                 iterations=1, isovalue=0.5, blur=0.0, debug=False):
        arguments = [os.path.join(dir, pveins_file)]
        arguments.append(os.path.join(dir, 'segmentation.s.nii'))
        arguments.append('-iterations')
        arguments.append(str(iterations))
        seg_1_out, _ = self.run_cmd(self._mirtk,'close-image', arguments, debug)

        if seg_1_out != 0:
            logger.error('Error in close image')

        arguments.clear()
        arguments = [os.path.join(dir, 'segmentation.s.nii')]
        arguments.append(os.path.join(dir, 'segmentation.vtk'))
        arguments.append('-isovalue')
        arguments.append(str(isovalue))
        arguments.append('-blur')
        arguments.append(str(blur))
        seg_2_out, _ = self.run_cmd(self._mirtk, 'extract-surface', arguments, debug)
        if seg_2_out != 0:
            logger.error('Error in extract surface')

        arguments.clear()
     
        arguments = [os.path.join(dir, 'segmentation.vtk')]
        arguments.append(os.path.join(dir, 'segmentation.vtk'))
        seg_5_out, _ = self.run_cmd(self._mirtk, 'smooth-surface', arguments, debug)
        if seg_5_out != 0:
            logger.error('Error in smooth image')
    
    def clip_mitral_valve(self, dir:str, pveins_file:str, mvi_name='prodMVI.vtk', debug=False) : 
        arguments = ['-i']
        arguments.append(os.path.join(dir, pveins_file))
        arguments.append('-mv')
        arguments.append('-mvname')
        arguments.append(mvi_name)

        mvi_out, _ = self.run_clip(arguments)
        if mvi_out != 0:
            logger.error('Error in clipping mitral valve')

    def create_scar_options_file(self, dir: str, opt_file='options.json', 
                                 output_dir='OUTPUT', legacy=False, limits=[-1, 3], radius=False, 
                                 method=1, threshold_values=[0.97, 1.2, 1.32] ) -> None:
        """Creates a basic file with scar options"""
        if method != 1 and method != 2:
            logger.error("Error: Method must be 1 (IIR) or 2 (Msd)")
            return 1
        
        dic = {
            "output_dir": output_dir,
            "roi_legacy_projection": legacy,
            "roi_limits": ','.join(map(str, limits)),
            "roi_radius": (radius or legacy),
            "threshold_values": ','.join(map(str, threshold_values)),
            "thresholds_method": method
        }

        output_path = os.path.join(dir, opt_file)
        logger.info(f'Creating options file: {output_path}')

        with open(output_path, 'w') as f:
            json.dump(dic, f)

    def create_scar_test_image(self, image_size, prism_size, method, origin, spacing, simple) :
        im, seg, boundic = itku.generate_scar_image(image_size, prism_size, origin, spacing, method, simple)
        return im, seg, boundic
    
    def get_threshold(self, method: int, value, mean_bp, std_bp):
        method_dict = {
            1 : lambda x, mbp, stdb : x*mbp,
            2 : lambda x, mbp, stdb : x*stdb + mbp
        }

        output = method_dict[method](value, mean_bp, std_bp) if value > 0 else 0
        return output
    
    def read_stats_from_file(self, file_path):
        mean_bp = None
        std_bp = None
        method = None 

        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        thresholds = []
        scores = []
        for ix, line in enumerate(lines):
            line = line.strip()  # Remove leading/trailing whitespace and newline characters
            if ix == 0:
                method = 1 if line.startswith("IIR_") else 2
                
            elif line.startswith("V="):
                # Extract threshold (V value) and add it to the list
                parts = line.split(", ")
                if len(parts) == 2 :
                    if parts[0].startswith("V="):
                        threshold = float(parts[0].split('=')[1])
                        thresholds.append(threshold)
                    if parts[1].startswith("SCORE="):
                        score = float(parts[1].split('=')[1])
                        scores.append(score)
            elif mean_bp is None:
                # The first two non-empty lines should be mean_bp and std_bp
                mean_bp = float(line)
            elif std_bp is None:
                std_bp = float(line)
        
        return method, mean_bp, std_bp, thresholds, scores
    
    def get_bloodpool_stats_from_file(self, file_path):
        _, mean_bp, std_bp, _, _ = self.read_stats_from_file(file_path)
        return mean_bp, std_bp
    
    def mask_voxels_above_threshold(self, im, mask, thres_mean, thres_std, thres_value=0, mask_value=0, ignore_im=None):
        thres = self.get_threshold(self.get_scar_method(), thres_value, thres_mean, thres_std)
        masked_im = itku.mask_image(im=im, mask=mask, mask_value=mask_value, threshold=thres, ignore_im=ignore_im)

        return masked_im
    
    def mask_segmentation_above_threshold(self, seg_path, im, mask, thres_mean, thres_std, thres_value=0, mask_value=0, ignore_im=None):
        thres = self.get_threshold(self.get_scar_method(), thres_mean, thres_std, thres_value)
        mask_from_im = itku.get_mask_with_restrictions(im, mask, thres, ignore_im=ignore_im)

        seg = itku.load_image(seg_path)
        masked_seg = itku.simple_mask(im=seg, mask=mask_from_im, mask_value=mask_value)
        return masked_seg

    def save_state(self, dir, fname) : 
        state_available = self.load_state(dir, fname)
        if state_available:
            logger.info(f'State file {fname} already exists. Overwriting...')

        state = {
            'cemrg' : {sys.platform: self.cemrg},
            'mirtk' : {sys.platform: self.mirtk},
            'scar_cmd_name' : self.scar_cmd_name,
            'clip_cmd_name' : self.clip_cmd_name,
        }
        with open(os.path.join(dir, fname), 'w') as f:
            json.dump(state, f)

    def load_state(self, dir, fname) : 
        if not os.path.isfile(os.path.join(dir, fname)):
            return False
        
        try:
            with open(os.path.join(dir, fname), 'r') as f:
                state = json.load(f)
        except:
            return False
        
        self.cemrg = state['cemrg'][sys.platform]
        self.mirtk = state['mirtk'][sys.platform]
        self.scar_cmd_name = state['scar_cmd_name']
        self.clip_cmd_name = state['clip_cmd_name']

        return True
