#경로
root_path = r'E:\2_SET_LEVEL\Q8\VerticalDivide\Final0109_Full'

# 해상도
h = 2504
v = 2256

# #Freq Code
# freq_name = ['120_HS', '10_HS', '1_HS'] # OO_HS, OO_NS 형식 입력, 순서 상관 없음
# freq_code = ['FUN/HS_120HZ_CNG', 'FUN/HS_10HZ_CNG',  'FUN/HS_1HZ_CNG']

# #Dim Code
dim_name = ['1500','500','183','98','20','1'] 
dim_code = ['WREG0 = 0x39, 0x51, 0x0D, 0xFF',
            'WREG0 = 0x39, 0x51, 0x07, 0xFF',
            'WREG0 = 0x39, 0x51, 0x02, 0xED',
            'WREG0 = 0x39, 0x51, 0x01, 0x91',
            'WREG0 = 0x39, 0x51, 0x00, 0x51',
            'WREG0 = 0x39, 0x51, 0x00, 0x04']

# #Crop Mask Define을 위한 최대 휘도
# crop_dim_code = ['WREG0 = 0x39, 0x51, 0x0D, 0xFF']  #HBM or Max Dim Code


freq_name = ['120_HS', '10_HS']  #, '1_HS']
freq_code = ['FUN/HS_120HZ_CNG' 'FUN/HS_10HZ_CNG']  #, 'FUN/HS_1HZ_CNG']
# dim_name = ['500'] 
# dim_code = ['WREG0 = 0x39, 0x51, 0x07, 0xFF']
crop_dim_code = ['WREG0 = 0x39, 0x51, 0x07, 0xFF']  #HBM or Max Dim Code

#AOD Mode
aod_mode = 0 # 0: 없음, 1: 있음
aod_freq_name = [30] #AOD 주파수, 숫자만 입력
aod_on2 = ['FUN/HLPM_HIGH_MODE']
aod_off = ['FUN/ALPM_SEQ_OFF']  

#촬상 Image Rotation
rot_degree = 270 #이미지 회전 각도 입력

#25℃ 설정 여부
temp_set = 0 # 0: 온도(25℃) 설정 안함, 1: 온도설정 함
temp_set_time = [10] #25℃ 설정까지 대기 시간(s)

#--------------------------------------------------------------#
AI = 1  # 0: 미실행, 1: 실행

primaries = {
    "W": (0.305, 0.321),  # 백색
    "R": (0.690, 0.309),  # 적색
    "G": (0.209, 0.748),  # 녹색
    "B": (0.142, 0.042),  # 청색
}


###################↑↑↑↑↑↑↑수정할것↑↑↑↑↑↑#######################


#import imp
import os
import time
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
from datetime import datetime
from datetime import timedelta
import cv2
from PIL import Image, ImageDraw, ImageFont
import random
import copy
import xlwings as xw # ver3.0 추가
import shutil # ver3.0 추가
import sys 
import math
import openpyxl as xl
from skimage import io, util
from Libs.image_process import Process
from Libs.image_process import Uniformity
import logging
import pyvisa as visa
import pandas as pd
import numpy as np
import threading
import ctypes
Thread = threading.Thread

log_path = r'D:\Platform\Log'

#############Temp Control################
class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                 args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        # print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)
            
    def join(self, *args):
        Thread.join(self, *args)
        return self._return


    def get_id(self): 
        # returns id of the respective thread 
        if hasattr(self, '_thread_id'): 
            return self._thread_id 
        for id, thread in threading._active.items(): 
            if thread is self: 
                return id
   

    def raise_exception(self): 
        thread_id = self.get_id() 
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 
              ctypes.py_object(SystemExit)) 
        if res > 1: 
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0) 
            print('Exception raise failure') 


STX = '\x02'

class Temp2500:
    __version__ = '25.8.8'  # 기본 기능 구현 및 초도 배포

    def __init__(self):
        self.comm = None
        self.rm = None
        self.logger = None
        try:
            self.rm = visa.ResourceManager()
        except Exception:
            try:
                self.rm = visa.ResourceManager(r'C:\Windows\System32\nivisa64.dll')
            except Exception:
                pass

    def __del__(self):
        if type(self.comm) is type(None): return
        self.comm.close()


    def _errer_msg(self):
        if not self.logger:
            self.logger = self.create_logger('temi2500')

        self.logger.error('Check NI-Visa driver is installed or not.')
        self.logger.error('Download URL : https://www.ni.com/ko-kr/support/downloads/drivers/download.ni-visa.html')


    def create_logger(self, logger_name):
        # Create Logger
        logger = logging.getLogger(logger_name)
     
        # Check handler exists
        if len(logger.handlers) > 0:
            return logger # Logger already exists
     
        logger.setLevel(logging.DEBUG)
     
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Create Handlers
        # stream_handler = logging.StreamHandler()
        # stream_handler.setFormatter(formatter)
        # logger.addHandler(stream_handler)
    
        file_name = time.strftime('%Y-%m-%d', time.localtime(time.time())) + f'_{logger_name}.log'
        file_handler = logging.FileHandler(os.path.join(log_path, file_name))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
        logger.info(f'{logger_name} Library start')
        return logger

    def get_port_list(self):
        if not self.logger:
            self.logger = self.create_logger('temi2500')

        self.logger.debug(f'[+] get_port_list')
        if type(self.rm) is type(None):
            self._errer_msg()
            return list()
        else:
            self.port_list = self.rm.list_resources()
            self.logger.debug(f'[-] get_port_list - {self.port_list}')
            return self.port_list


    def connect(self, port_name, baud_rate=38400,
                data_bits=8, parity=0, stop_bits=10,
                flow_control=0, timeout=3000,
                write_termination='\r\n',
                read_termination='\r\n'):

        """
        inputs
            - port_name (str) : get_port_list() 호출 후 string 으로 입력 또는 COM3, COM4 와 같이 입력
            - baud_rate (int) : 9600 (장비 설정에 따라)
            - data_bits (int) : 8 / 7
            - parity (int) : 0(none), 1(odd), 2(even)
            - stop_bits (int) : 10(1bit) / 15(1.5bit) / 20(2bit)
            - flow_control (int) : 0(none) / 1(xon_xoff) /2(rts_cts)
            - timeout (int) : milisec
        """
        if not self.logger:
            self.logger = self.create_logger('temp2500')

        self.logger.debug(f'[+] connect - {port_name}')
        port_name = port_name.upper()
        if 'ASRL' in port_name:
            port_name = port_name.replace('ASRL', 'COM').split(':')[0]

        self.port_name = port_name
        print('port_name : ', port_name)
        
        try:
            session = self.rm.session
        except Exception:
            self.rm = visa.ResourceManager()
            
        try:
            self.comm = self.rm.open_resource(self.port_name, baud_rate=baud_rate, data_bits=data_bits,
                                              parity=parity, stop_bits=stop_bits,
                                              flow_control=flow_control, timeout=timeout,
                                              write_termination=write_termination,
                                              read_termination=read_termination
                                              )
            ret = self.check_status()[0]
            self.logger.debug(f'[-] connect - {ret}')
            return ret
        except Exception as e:
            self.logger.error(f'{e}')
            return False


    def disconnect(self):
        if not self.logger:
            self.logger = self.create_logger('temi2500')

        self.logger.debug(f'[+] disconnect')
        if type(self.comm) is type(None): return
        try:
            self.is_auto_update = False
            self.comm.close()
            self.comm = None
            self.rm.close()
            self.logger.debug(f'[-] disconnect')
            return True
        except Exception as e:
            self.logger.error(f'{e}')
            self.comm = None
            return False
        
        
        
    def get_checksum(self, txt):
        data = 0
        txt = txt.encode('cp949')
        for i in txt:
            data += i
        
        checksum = data % 256
        
        return f'{checksum:02X}'
        


    def start_auto_update(self, interval_sec=2.5, signal_draw=None):
        self.logger.debug(f'[+] start_auto_update')
        self.df = pd.DataFrame()
        self.is_auto_update = True
        self.t = ThreadWithReturnValue(target=self.auto_update, name='TEMI200:auto_update', args=(interval_sec, signal_draw))
        self.t.start()
        self.logger.debug(f'[-] start_auto_update')


    def stop_auto_update(self):
        self.logger.debug(f'[+] stop_auto_update')
        self.is_auto_update = False
        if not self.t:
            return
        if self.t.is_alive():
            self.t.raise_exception()
        self.logger.debug(f'[-] stop_auto_update')


    def auto_update(self, interval_sec, signal_draw):
        while(self.is_auto_update):
            sr = self.measure()
            self.df = self.df.append(sr, ignore_index=True)
            if signal_draw:
                signal_draw.emit(self.df)
            time.sleep(interval_sec)
            

    def check_status(self):
        if type(self.comm) is type(None): return
        ret = self.query(STX + '01RSD,01,0101')
        return self._parse_return_status(ret)

    def write(self, command):
        """
        inputs
            - command : str
        """
        try:
            if type(self.comm) is type(None): return
            self.comm.write(command)
        except Exception as e:
            self.logger.error(f'{e}')    

    def read(self):
        """
        outputs
            - return : str
        """
        try:
            if type(self.comm) is type(None): return
            ret = self.comm.read()
            return ret
        except Exception as e:
            self.logger.error(f'{e}')    

    def query(self, command):
        """
        inputs
            - command : str
        outputs
            - return : str
        """
        try:
            if type(self.comm) is type(None): return
            ret = self.comm.query(command)
            return ret
        except Exception as e:
            self.logger.error(f'{e}')
        

    def run(self):
        self.logger.debug(f'[+] run')
        if type(self.comm) is type(None): return
        
        command = '01WSD,01,0102,0001'
        cs = self.get_checksum(command)
        ret = self.query(STX + command + cs)
        data = self._parse_return_status(ret)
        self.logger.debug(f'[-] run - {data}')
        return data
    

    def stop(self):
        self.logger.debug(f'[+] stop')
        if type(self.comm) is type(None): return
        
        command = '01WSD,01,0102,0004'
        cs = self.get_checksum(command)
        ret = self.query(STX + command + cs)
        data = self._parse_return_status(ret)
        self.logger.debug(f'[-] stop - {data}')
        return data

    def set_temp(self, temp):
        """
        inputs
            - temp : float (타켓 온도 설정)
        """
        self.logger.debug(f'[+] set_temp - {temp}')
        if type(self.comm) is type(None): return
        str_temp = f'{np.uint16(np.int16(temp * 10)):04X}'
        
        command = f'01WSD,01,0104,{str_temp}'
        cs = self.get_checksum(command)
        ret = self.query(STX + command + cs)
        data = self._parse_return_status(ret)
        self.logger.debug(f'[-] set_temp - {data}')
        return data


    def measure(self):
        """
        outputs
           (pandas Series)
        """
        self.logger.debug(f'[+] measure')
        if type(self.comm) is type(None): return
                
        command = '01RSD,04,0001'
        cs = self.get_checksum(command)
        ret = self.query(STX + command + cs)

        values = self._parse_data(ret)
        index = ['temp_now', 'temp_target']
        # print(f'values: {values}')
        sr = pd.Series([values[0], values[2]], index)
        self.logger.info(sr)
        self.logger.debug(f'[-] measure')
        return sr

    def _parse_return_status(self, ret):
        ret = ret.replace(STX, '')
        if 'ok' in ret.lower():
            return [True, ret]
        else:
            return [False, ret]

    def _parse_data(self, data):
        values = None
        try:
            data_list = data.split(',')
            print(data_list)
            values = list(map(self._hex2float, data_list[2:5]))
            print(values)
        except Exception as e:
            self.logger.error(f'{e}')
        return values

    def _hex2float(self, data):
        value = int(data, 16)
        value = np.int16(np.uint16(value))
        value /= 10
        return value

if __name__ == '__main__':
    temp = Temp2500()
    # Ca310_1 = CaX10Lib()

    port_list = temp.get_port_list()
    print(port_list)

    ret = temp.connect('COM15')

      
if temp_set == 0:  
    print("\n<<<<<온도 설정 안함>>>>>>\n")
        
elif temp_set == 1:  

    time.sleep(1)
    temp.set_temp(25.0)
    temp.measure()
    temp.run()
    time.sleep(temp_set_time[0]) 
    
   


#################################################################
# 프로그램 시작 시간 기록
start_time = time.time()
log_path = r'D:\Platform\Log'

ip = Process()
uf = Uniformity()

filename = os.path.join(root_path, "2D.txt")
path_save = f'{root_path}'
if not os.path.exists(path_save):
    os.makedirs(path_save)
    
def print_log(log):
    print(log)
    with open(filename, 'a') as f:
        f.write(log + '\n')

# 계측 Point param
threshold = 0.6

print_log("\n\nWecome!!!\n")

# Crop Mask
TxHost.send_command(freq_code[0])
TxHost.send_command(crop_dim_code[0])
TxHost.send_command("pattern0 = 255,1,1", True, 1000)
TxHost.send_command("DELAYMS = 500", True, 1000)
M631.set_param(is_hybrid_mode=1, f_cam_exp=16.666, f_spm_exp=16.666, n_cam_avg = 1, n_spm_avg = 1, nd_position = 1)
M631.grab_only()
white_img_mask = M631.get_xyz()
mask_Y = white_img_mask[:][:][1]
[crop_mask_Y, pos, _] = ip.auto_crop(mask_Y, threshold, h, v, rot_degree, draw=True)

print_log("\nCrop mask is ready.\n")
print_log("Starting measuremnt!\n")
print_log(str(datetime.now()))

#Image Code
# img_code = [f'PTRN_SET = PaintXY, 0,  0, {h}, {v}, 0x00FFFFFF\nPTRN_SET = Write, 0',
            # f'PTRN_SET = PaintXY, 0,  0, {h}, {v}, 0x0000FF00\nPTRN_SET = Write, 0',
            # f'PTRN_SET = PaintXY, 0,  0, {h}, {v}, 0x00A0A0A0\nPTRN_SET = Write, 0',
            # f'PTRN_SET = PaintXY, 0,  0, {h}, {v}, 0x00808080\nPTRN_SET = Write, 0',
            # f'PTRN_SET = Checker, 0, 0, {h}, {v}, {h}, {v}, 0x00FFFFFF, 0x00000000\nPTRN_SET= Write, 0'',
            # f'PTRN_SET = Checker, 0, 0, {h}, {v}, {int(h//2)}, {int(v//2)}, 0x00FFFFFF, 0x00000000\nPTRN_SET= Write, 0'',          
            # f'PTRN_SET = Zebra4H, 0, 0, {h}, {v}, 1, 0x00000000, 0x00FFFFFF\nPTRN_SET= Write, 0',
            # f'PTRN_SET = Zebra4H, 0, 0, {h}, {v}, 2, 0x00000000, 0x00FFFFFF\nPTRN_SET= Write, 0',
            # f'PTRN_SET = VLine, 0, 0, {h}, {v}, 0x00FFFFFF, 0x00000000\nPTRN_SET= Write, 0',
            # f'PTRN_SET = Checker, 0, 0, {h}, {v}, {int(h//2)}, 1, 0x00FFFFFF, 0x00000000\nPTRN_SET= Write, 0'',
            # ]
# imgs_name = ['W255',
             # 'G255',
             # 'W160',
             # 'W128'
             # '1-Dot',
             # '2-Dot',
             # '1H',
             # '2H',
             # '1V',
             # '2V'
             # ]

img_code = [f'PTRN_SET = IMG_READ, {h}, {v}, t2_10_h.png\nPTRN_SET = Write, 0',
              f'PTRN_SET = PaintXY, 0,  0, {h}, {v}, 0x00FFFFFF\nPTRN_SET = Write, 0',
              f'PTRN_SET = PaintXY, 0,  0, {h}, {v}, 0x00A4A4A4\nPTRN_SET = Write, 0',
              f'PTRN_SET = PaintXY, 0,  0, {h}, {v}, 0x00484848\nPTRN_SET = Write, 0',
              f'PTRN_SET = PaintXY, 0,  0, {h}, {v}, 0x00303030\nPTRN_SET = Write, 0',
              f'PTRN_SET = PaintXY, 0,  0, {h}, {v}, 0x00181818\nPTRN_SET = Write, 0',
              f'PTRN_SET = PaintXY, 0,  0, {h}, {v}, 0x00101010\nPTRN_SET = Write, 0',
              f'PTRN_SET = PaintXY, 0,  0, {h}, {v}, 0x0000FF00\nPTRN_SET = Write, 0',
              f'PTRN_SET = PaintXY, 0,  0, {h}, {v}, 0x00A0A0A0\nPTRN_SET = Write, 0',
              f'PTRN_SET = PaintXY, 0,  0, {h}, {v}, 0x007F7F7F\nPTRN_SET = Write, 0',
              f'PTRN_SET = VScaleFull, 0,  0, {h}, {v}, 1,1,1\nPTRN_SET= Write, 0',
              f'PTRN_SET = VScaleFull, 0,  0, {h}, {v}, -1,-1,-1\nPTRN_SET= Write, 0',
              f'PTRN_SET = IMG_READ, {h}, {v}, t2_4_d.png\nPTRN_SET = Write, 0',
              f'PTRN_SET = IMG_READ, {h}, {v}, t2_4_e.png\nPTRN_SET = Write, 0',
              f'PTRN_SET = IMG_READ, {h}, {v}, t2_4_f.png\nPTRN_SET = Write, 0',
              f'PTRN_SET = IMG_READ, {h}, {v}, t2_4_g.png\nPTRN_SET = Write, 0',
              f'PTRN_SET = IMG_READ, {h}, {v}, t2_4_h.png\nPTRN_SET = Write, 0',
              f'PTRN_SET = IMG_READ, {h}, {v}, t2_4_i.png\nPTRN_SET = Write, 0',
              f'PTRN_SET = IMG_READ, {h}, {v}, t2_5_c.png\nPTRN_SET = Write, 0',
              f'PTRN_SET = IMG_READ, {h}, {v}, t2_10_a.png\nPTRN_SET = Write, 0',
              f'PTRN_SET = IMG_READ, {h}, {v}, t2_10_b.png\nPTRN_SET = Write, 0',
              f'PTRN_SET = IMG_READ, {h}, {v}, t2_10_c.png\nPTRN_SET = Write, 0',
              f'PTRN_SET = IMG_READ, {h}, {v}, t2_10_d.png\nPTRN_SET = Write, 0',
              f'PTRN_SET = IMG_READ, {h}, {v}, t2_10_e.png\nPTRN_SET = Write, 0',
              f'PTRN_SET = IMG_READ, {h}, {v}, t2_10_f_.png\nPTRN_SET = Write, 0',
              f'PTRN_SET = IMG_READ, {h}, {v}, t2_10_g.png\nPTRN_SET = Write, 0',
              f'PTRN_SET = IMG_READ, {h}, {v}, t2_10_i.png\nPTRN_SET = Write, 0',
              f'PTRN_SET = IMG_READ, {h}, {v}, t2_10_k.png\nPTRN_SET = Write, 0',
              f'PTRN_SET = IMG_READ, {h}, {v}, t2_10_a_.png\nPTRN_SET = Write, 0',
              f'PTRN_SET = IMG_READ, {h}, {v}, t2_10_b_.png\nPTRN_SET = Write, 0',
              f'PTRN_SET = IMG_READ, {h}, {v}, t2_10_c_.png\nPTRN_SET = Write, 0',
              f'PTRN_SET = IMG_READ, {h}, {v}, t2_10_f.png\nPTRN_SET = Write, 0',
              f'PTRN_SET = IMG_READ, {h}, {v}, t2_10_k_.png\nPTRN_SET = Write, 0',
              f'PTRN_SET = IMG_READ, {h}, {v}, t2_14_c.png\nPTRN_SET = Write, 0',
              f'PTRN_SET = IMG_READ, {h}, {v}, t2_14_d.png\nPTRN_SET = Write, 0',
              f'PTRN_SET = IMG_READ, {h}, {v}, t2_14_e.png\nPTRN_SET = Write, 0',
              f'PTRN_SET = IMG_READ, {h}, {v}, t2_14_f.png\nPTRN_SET = Write, 0',
              f'PTRN_SET = IMG_READ, {h}, {v}, t2_16_c_.png\nPTRN_SET = Write, 0',
              f'PTRN_SET = IMG_READ, {h}, {v}, t2_16_d_.png\nPTRN_SET = Write, 0',
              f'PTRN_SET = IMG_READ, {h}, {v}, t2_16_e_.png\nPTRN_SET = Write, 0',
              f'PTRN_SET = IMG_READ, {h}, {v}, t2_18_b.png\nPTRN_SET = Write, 0',
              f'PTRN_SET = IMG_READ, {h}, {v}, t2_18_c.png\nPTRN_SET = Write, 0',
              f'PTRN_SET = IMG_READ, {h}, {v}, t2_18_d.png\nPTRN_SET = Write, 0',
              f'PTRN_SET = IMG_READ, {h}, {v}, t2_18_e.png\nPTRN_SET = Write, 0',
              f'PTRN_SET = IMG_READ, {h}, {v}, t2_18_f.png\nPTRN_SET = Write, 0',
              f'PTRN_SET = IMG_READ, {h}, {v}, t2_18_g.png\nPTRN_SET = Write, 0',
              f'PTRN_SET = IMG_READ, {h}, {v}, t2_18_h.png\nPTRN_SET = Write, 0',
              f'PTRN_SET = IMG_READ, {h}, {v}, t2_18_i.png\nPTRN_SET = Write, 0',
              f'PTRN_SET = IMG_READ, {h}, {v}, t2_18_j.png\nPTRN_SET = Write, 0',
              f'PTRN_SET = IMG_READ, {h}, {v}, t2_18_k.png\nPTRN_SET = Write, 0',
              f'PTRN_SET = HLine, 0, 0, {h//2}, {v}, 0x00FFFFFF, 0x00000000\nPTRN_SET = HLine, {h//2}, 0, {h//2}, {v}, 0x00000000, 0x00FFFFFF\nPTRN_SET= Write, 0', 
              f'PTRN_SET = HLine, 0, 0, {h//2}, {v}, 0x007F7F7F, 0x00000000\nPTRN_SET = HLine, {h//2}, 0, {h//2}, {v}, 0x00000000, 0x007F7F7F\nPTRN_SET= Write, 0',   
              f'PTRN_SET= VScaleFull, 0,  0,' + str(h) + ',' + str(v) + ', 1, 0, 0\nPTRN_SET= Write, 0',
              f'PTRN_SET= VScaleFull, 0,  0,' + str(h) + ',' + str(v) + ', 0, 1, 0\nPTRN_SET= Write, 0',
              f'PTRN_SET= VScaleFull, 0,  0,' + str(h) + ',' + str(v) + ', 0, 0, 1\nPTRN_SET= Write, 0',         
              f'PTRN_SET= VScaleFull, 0,  0,' + str(h) + ',' + str(v) + ', -1, 0, 0\nPTRN_SET= Write, 0',
              f'PTRN_SET= VScaleFull, 0,  0,' + str(h) + ',' + str(v) + ', 0, -1, 0\nPTRN_SET= Write, 0',
              f'PTRN_SET= VScaleFull, 0,  0,' + str(h) + ',' + str(v) + ', 0, 0, -1\nPTRN_SET= Write, 0',
              f'PTRN_SET = Zebra4H, 0, 0, {h}, {v}, 5, 0x00000000, 0x00FFFFFF\nPTRN_SET= Write, 0',
              f'PTRN_SET = Zebra4H, 0, 0, {h}, {v}, 4, 0x00000000, 0x00FFFFFF\nPTRN_SET= Write, 0',
              f'PTRN_SET = Zebra4H, 0, 0, {h}, {v}, 3, 0x00000000, 0x00FFFFFF\nPTRN_SET= Write, 0',
              f'PTRN_SET = Zebra4H, 0, 0, {h}, {v}, 2, 0x00000000, 0x00FFFFFF\nPTRN_SET= Write, 0',
              f'PTRN_SET = Zebra4H, 0, 0, {h}, {v}, 1, 0x00000000, 0x00FFFFFF\nPTRN_SET= Write, 0',
              f'PTRN_SET = VLine, 0, 0, {h}, {v}, 0x00FFFFFF, 0x00000000\nPTRN_SET= Write, 0',
              f'PTRN_SET = Checker, 0, 0, {h}, {v}, {h}, {v}, 0x00FFFFFF, 0x00000000\nPTRN_SET= Write, 0',
              f'PTRN_SET = IMG_READ, {h}, {v}, t2_29_i.png\nPTRN_SET = Write, 0']
            

                       
imgs_name = ['t2_10_h',
             'W255', 'W164', 'W72', 'W48', 'W24', 'W16', 
             'G255', 'W160', 'W127', 
             'VScale_255_0', 'VScale_0_255', 
             't2_4_d', 't2_4_e', 't2_4_f', 't2_4_g', 't2_4_h', 't2_4_i', 
             't2_5_c',
             't2_10_a', 't2_10_b', 't2_10_c', 't2_10_d','t2_10_e', 't2_10_f_', 't2_10_g', 't2_10_i','t2_10_k',
             't2_10_a_', 't2_10_b_', 't2_10_c_', 't2_10_f', 't2_10_k_',
             't2_14_c', 't2_14_d', 't2_14_e', 't2_14_f', 
             't2_16_c_','t2_16_d_','t2_16_e_',
             't2_18_b', 't2_18_c', 't2_18_d', 't2_18_e', 't2_18_f', 't2_18_g', 't2_18_h', 't2_18_i', 't2_18_j', 't2_18_k',
             'E_O_255', 'E_O_127',
             'VScale_R_255_0','VScale_G_255_0','VScale_B_255_0','VScale_R_0_255','VScale_G_0_255','VScale_B_0_255',
             '5H', '4H','3H','2H','1H', '1V', '1dot','2x1V']


AOD_img_code = [f'PTRN_SET = IMG_READ, {h}, {v}, t2_16_f.png\nPTRN_SET = Write, 0',
                f'PTRN_SET = IMG_READ, {h}, {v}, t2_16_g.png\nPTRN_SET = Write, 0',
                f'PTRN_SET = IMG_READ, {h}, {v}, t2_16_h.png\nPTRN_SET = Write, 0']
AOD_imgs_name = ['t2_16_f','t2_16_g','t2_16_h']


def Image_grab(freq_name, i, dims_name, img_name, pos, h, v, rot_degree, path_save):
    print_log("\nFrequency: %-10s Dimming: %-10s Image: %-10s "%(str(freq_name[i]), dims_name, img_name))
    print_log(str(datetime.now()))
    freq_values = [int(item.split('_')[0]) for item in freq_name]
    M631.set_cam_auto_exp(True, 60, 5, 200000, True, freq_values[i]) # cam_auto_exp, target, tol, max_exp, freq_enable, freq)  
    M631.set_spm_auto_exp(True, 60, 5, 200000, True, freq_values[i]) # spm_auto_exp, target, tol, max_exp, freq_enable, freq)
    M631.grab_only()
    [X,Y,Z, result] = M631.get_xyz()
    
    X_crop = ip.mask_crop(X, pos, h, v, rot_degree, draw=False)
    Y_crop = ip.mask_crop(Y, pos, h, v, rot_degree, draw=True)
    Z_crop = ip.mask_crop(Z, pos, h, v, rot_degree, draw=False)
    
#    X_out =  cv2.medianBlur(np.array(X_crop, dtype='float32'),5)
#    Y_out =  cv2.medianBlur(np.array(Y_crop, dtype='float32'),5)
#    Z_out =  cv2.medianBlur(np.array(Z_crop, dtype='float32'),5)
    
    print_log("Saving CSV files")      
    fname = str(img_name) + " " + str(freq_name[i]) + " " + str(dims_name)
    if os.path.isfile(path_save + '\\' + fname + '.mim'):          
        os.remove(path_save + '\\' + fname + '.mim')
    io.imsave(path_save + '\\' + fname + '.tiff', Y_crop)
    os.rename(path_save + '\\' + fname + '.tiff', path_save + '\\' + fname + '.mim')
    time.sleep(1)
#    np.savetxt(path_save + '\\' + fname + '_X_medianblur.csv', np.array(X_out), delimiter=",", fmt="%.2f") 
#    np.savetxt(path_save + '\\' + fname + '_Y_medianblur.csv', np.array(Y_out), delimiter=",", fmt="%.2f") 
#    np.savetxt(path_save + '\\' + fname + '_Z_medianblur.csv', np.array(Z_out), delimiter=",", fmt="%.2f") 
    
#    np.savetxt(path_save + '\\' + fname + '_X.csv', np.array(X_crop), delimiter=",", fmt="%.2f") 
#    np.savetxt(path_save + '\\' + fname + '_Y.csv', np.array(Y_crop), delimiter=",", fmt="%.2f") 
#    np.savetxt(path_save + '\\' + fname + '_Z.csv', np.array(Z_crop), delimiter=",", fmt="%.2f") 
    
    Y_norm = (Y_crop - Y_crop.min()) / (Y_crop.max() - Y_crop.min())
    #Y_norm = Y_crop / float(dims_name)
    Y_uint8 = util.img_as_ubyte(Y_norm)
    io.imsave(path_save + '\\' + fname + '_Y.png', Y_uint8)
    time.sleep(1)
    
    X_f16 = np.clip(X_crop, 0, None).astype(np.float16)
    Y_f16 = np.clip(Y_crop, 0, None).astype(np.float16)
    Z_f16 = np.clip(Z_crop, 0, None).astype(np.float16)
    XYZ_f16 = np.stack([X_f16, Y_f16, Z_f16], axis=-1)
    np.savez_compressed(path_save + '\\' + fname + '_f16.npz', data=XYZ_f16)
    time.sleep(1)
#    X_f32 = np.clip(X_crop, 0, None).astype(np.float32)
#    Y_f32 = np.clip(Y_crop, 0, None).astype(np.float32)
#    Z_f32 = np.clip(Z_crop, 0, None).astype(np.float32)
#    XYZ_f32 = np.stack([X_f32, Y_f32, Z_f32], axis=-1)
#    time.sleep(1)
#    np.savez_compressed(path_save + '\\' + fname + '_f32.npz', data=XYZ_f32)

    print_log(f'signal_level : {result.cam_sig_level_xyz.contents[:]}')
    print_log(f'cam_exposure_time : {result.cam_exp_time}')
    print_log(f'spm_exposure_time : {result.spm_exp_time}')
    time.sleep(1)
    
def Image_grab_HLPM(AOD_imgs_name, pos, h, v, rot_degree, path_save):
    TxHost.send_command(aod_on2[0])
    time.sleep(0.5)
    #print_log("\nAOD  Dimming: %-10s Image: %-10s "%(dims_name, img_name))
    print_log(str(datetime.now()))
    M631.set_cam_auto_exp(True, 60, 5, 200000, True, aod_freq_name[0]) # cam_auto_exp, target, tol, max_exp, freq_enable, freq)  
    M631.set_spm_auto_exp(True, 60, 5, 200000, True, aod_freq_name[0]) # spm_auto_exp, target, tol, max_exp, freq_enable, freq)
    M631.grab_only()
    [X,Y,Z, result] = M631.get_xyz()
    
    X_crop = ip.mask_crop(X, pos, h, v, rot_degree, draw=False)
    Y_crop = ip.mask_crop(Y, pos, h, v, rot_degree, draw=True)
    Z_crop = ip.mask_crop(Z, pos, h, v, rot_degree, draw=False)
    
#    X_out =  cv2.medianBlur(np.array(X_crop, dtype='float32'),5)
#    Y_out =  cv2.medianBlur(np.array(Y_crop, dtype='float32'),5)
#    Z_out =  cv2.medianBlur(np.array(Z_crop, dtype='float32'),5)
    
    print_log("Saving CSV files")      
    fname = str(img_name) + "_AOD" 
    if os.path.isfile(path_save + '\\' + fname + '.mim'):          
        os.remove(path_save + '\\' + fname + '.mim')
    io.imsave(path_save + '\\' + fname + '.tiff', Y_crop)
    os.rename(path_save + '\\' + fname + '.tiff', path_save + '\\' + fname + '.mim')
    time.sleep(1)
#    np.savetxt(path_save + '\\' + fname + '_X_mediablur.csv', np.array(X_out), delimiter=",", fmt="%.2f") #Medianblur
#    np.savetxt(path_save + '\\' + fname + '_Y_mediablur.csv', np.array(Y_out), delimiter=",", fmt="%.2f") #Medianblur
#    np.savetxt(path_save + '\\' + fname + '_Z_mediablur.csv', np.array(Z_out), delimiter=",", fmt="%.2f") #Medianblur
    
#    np.savetxt(path_save + '\\' + fname + '_X.csv', np.array(X_crop), delimiter=",", fmt="%.2f") 
#    np.savetxt(path_save + '\\' + fname + '_Y.csv', np.array(Y_crop), delimiter=",", fmt="%.2f") 
#    np.savetxt(path_save + '\\' + fname + '_Z.csv', np.array(Z_crop), delimiter=",", fmt="%.2f") 
    
    Y_norm = (Y_crop - Y_crop.min()) / (Y_crop.max() - Y_crop.min())
    #Y_norm = Y_crop / float(dims_name)
    Y_uint8 = util.img_as_ubyte(Y_norm)
    io.imsave(path_save + '\\' + fname + '_Y.png', Y_uint8)
    time.sleep(1)
    
    X_f16 = np.clip(X_crop, 0, None).astype(np.float16)
    Y_f16 = np.clip(Y_crop, 0, None).astype(np.float16)
    Z_f16 = np.clip(Z_crop, 0, None).astype(np.float16)
    XYZ_f16 = np.stack([X_f16, Y_f16, Z_f16], axis=-1)
    np.savez_compressed(path_save + '\\' + fname + '_f16.npz', data=XYZ_f16)
    time.sleep(1)
#    X_f32 = np.clip(X_crop, 0, None).astype(np.float32)
#    Y_f32 = np.clip(Y_crop, 0, None).astype(np.float32)
#    Z_f32 = np.clip(Z_crop, 0, None).astype(np.float32)
#    XYZ_f32 = np.stack([X_f32, Y_f32, Z_f32], axis=-1)
#    time.sleep(1)
#    np.savez_compressed(path_save + '\\' + fname + '_f32.npz', data=XYZ_f32)

    print_log(f'signal_level : {result.cam_sig_level_xyz.contents[:]}')
    print_log(f'cam_exposure_time : {result.cam_exp_time}')
    print_log(f'spm_exposure_time : {result.spm_exp_time}')
    TxHost.send_command(aod_off[0])
    time.sleep(1)

# Measurement
#total = len(freq_code)*len(img_code)*len(dim_code)
#cnt = 1

freq_values = [int(item.split('_')[0]) for item in freq_name]

####################################################################################################################################################
#print_log(f'cam_exposure_time : {result.cam_exp_time}')
#print_log(f'spm_exposure_time : {result.spm_exp_time}')
#TxHost.key_reset()
#delayms = int(result.spm_exp_time)
#print("\n<<<<<RESET>>>>>>\n")
#TxHost.key_next()
#print("\n<<<<<NEXT>>>>>>\n")
#time.sleep(1.5)
####################################################################################################################################################

# 초기화
TxHost.send_command('FUN/GMA_CHOP_ON')
TxHost.send_command('FUN/GMA_AUTO_ZERO_OFF')
TxHost.send_command('FUN/SRC_COL_CHOP_ON')

for i in range(len(freq_code)):
    TxHost.send_command(freq_code[i])    
    if freq_values[i] == 1:
        time.sleep(0.5)
        for j in range(len(dim_code)):
            TxHost.send_command(dim_code[j])
            time.sleep(0.5)
            for k in range(len(img_code)):
                TxHost.send_command(img_code[k])
                time.sleep(3.5)
                img_name = imgs_name[k]
                dims_name = dim_name[j]
                Image_grab(freq_name, i, dims_name, img_name, pos, h, v, rot_degree, path_save)
                
    else:           
        time.sleep(0.5)
        for j in range(len(dim_code)):
            TxHost.send_command(dim_code[j])
            time.sleep(0.5)
            for k in range(len(img_code)):
                TxHost.send_command(img_code[k])
                time.sleep(1.5)
                img_name = imgs_name[k]
                dims_name = dim_name[j]
                Image_grab(freq_name, i, dims_name, img_name, pos, h, v, rot_degree, path_save)                

if aod_mode == 0:  #AOD Mode 없음
    print("\nAOD Mode 없음\n")

elif aod_mode == 1:  #AOD Mode 있음
    
    if aod_freq_name[0] == 1:
    
        img_name = AOD_imgs_name[0]
        TxHost.send_command(AOD_img_code[0])
        time.sleep(3.5)
        Image_grab_HLPM(AOD_imgs_name, pos, h, v, rot_degree, path_save)
    
        img_name = AOD_imgs_name[1]
        TxHost.send_command(AOD_img_code[1])
        time.sleep(3.5)
        Image_grab_HLPM(AOD_imgs_name, pos, h, v, rot_degree, path_save)

        img_name = AOD_imgs_name[2]
        TxHost.send_command(AOD_img_code[2])
        time.sleep(3.5)
        Image_grab_HLPM(AOD_imgs_name, pos, h, v, rot_degree, path_save)
        
    else:
        img_name = AOD_imgs_name[0]
        TxHost.send_command(AOD_img_code[0])
        time.sleep(0.5)
        Image_grab_HLPM(AOD_imgs_name, pos, h, v, rot_degree, path_save)
    
        img_name = AOD_imgs_name[1]
        TxHost.send_command(AOD_img_code[1])
        time.sleep(0.5)
        Image_grab_HLPM(AOD_imgs_name, pos, h, v, rot_degree, path_save)

        img_name = AOD_imgs_name[2]
        TxHost.send_command(AOD_img_code[2])
        time.sleep(0.5)
        Image_grab_HLPM(AOD_imgs_name, pos, h, v, rot_degree, path_save)    
    
M631.set_cam_auto_exp(enable=False) 
M631.set_spm_auto_exp(enable=False)    
#TxHost.send_command(f'pattern0 = 0,1,1')
TxHost.key_reset() 
time.sleep(1)

if temp_set == 0:  
    print_log("\n\nEND!!!\n")
        
elif temp_set == 1:  
    

    temp.stop()
    print_log("\nChamber Stop\n")
    temp.disconnect()
    print_log("\nChamber Disconnect\n")
    print_log("\n\nThe End.\n\n")


# AI 판정 시스템
if AI == 0:  #AI 판정 미실행
    print("\nAI 판정 미실행\n")

elif AI == 1:  #AI 판정 실행
    import os
    import json
    import subprocess
    from tqdm import tqdm
    import time
    import sys
    import re
    import traceback

    # JSON 문자열로 변환
    primaries_str = json.dumps(primaries)
    print(f" > primaries : {primaries_str} ")

    # 실행할 스크립트 경로
    script_path = r"D:\Non_Documents\_AI_NAMU\20251117_M631R\main_AI_20260106_17.py"
    cmd = ["python", script_path, root_path, primaries_str]

    print("\n  AI 추론 시작 ... ")

    # 실시간 진행률 바 (images 기반)
    pbar = None
    start_time = time.time()

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding='utf-8',
            errors='replace',
            universal_newlines=True
        )

        total_images = 0
        current_image = 0
        avg_time_per_image = 0.0
        estimated_remaining = 0

        pbar = None

        for line in process.stdout:
            try:
                line = line.rstrip()
                if line:
                    print(line)

                    # 전체 이미지 수 추출 (처음 한 번만)
                    if "Total images to process:" in line:
                        match = re.search(r"Total images to process:\s*(\d+)", line)
                        if match:
                            total_images = int(match.group(1))
                            pbar = tqdm(total=total_images, desc="진행률", unit="image", ascii=True, dynamic_ncols=True)

                    # 현재 이미지 처리 완료 추출
                    elif "[진행 중]" in line:
                        match = re.search(r'\[진행 중\]\s*\[(\d+)/(\d+)\]', line)
                        if match:
                            current, total = int(match.group(1)), int(match.group(2))
                            increment = current - current_image
                            if increment > 0 and pbar is not None:
                                elapsed = time.time() - start_time
                                avg_time_per_image = elapsed / current
                                estimated_remaining = avg_time_per_image * (total_images - current)
                                pbar.set_postfix_str(
                                    f"{avg_time_per_image:.2f}s/이미지, 남은 시간: {int(estimated_remaining // 60)}분 {int(estimated_remaining % 60)}초"
                                )
                                pbar.update(increment)
                                current_image = current

                    # 완료 처리
                    elif "# 추론 완료" in line or "추론 완료" in line:
                        if pbar is not None:
                            pbar.set_postfix_str("완료")
                            pbar.update(total_images - current_image)
                            current_image = total_images

            except Exception as e:
                print(f"[오류] 출력 처리 중 예외: {e}")
                continue

        process.wait()

        if process.returncode == 0:
            print("\n # AI 스크립트 정상 종료")
        else:
            print(f"\n # AI 스크립트 오류 발생: 반환 코드 {process.returncode}")

    except Exception as e:
        print(f"\n[오류] 실행 중 예외 발생: {e}")
        traceback.print_exc()
    finally:
        if pbar is not None:
            pbar.close()
        elapsed = time.time() - start_time
        print(f"\n -------  AI End  ------- ")
        print(f"총 소요 시간: {int(elapsed // 60)}분 {int(elapsed % 60)}초")


#-------------------------------------------------------------------------#    
# 프로그램 종료 시간 기록
end_time = time.time()

# 실행 시간 계산 (초 단위)
execution_time = int(end_time - start_time)

# 시간 단위로 변환
days = execution_time // 86400
hours = (execution_time % 86400) // 3600
minutes = (execution_time % 3600) // 60
seconds = execution_time % 60


# 조건에 따라 형식 지정
if days > 0:
    print(f"실행 시간: {days}일 {hours}시간 {minutes}분 {seconds}초")
elif hours > 0:
    print(f"실행 시간: {hours}시간 {minutes}분 {seconds}초")
elif minutes > 0:
    print(f"실행 시간: {minutes}분 {seconds}초")
else:
    print(f"실행 시간: {seconds}초")
    
    
