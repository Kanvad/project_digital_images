#-------------------------------
# imports
#-------------------------------

# builtins
import os
import sys
import time
import traceback
from math import hypot
from datetime import datetime
import json

# must be installed using pip
import numpy as np
import cv2

# local clayton libs
import frame_capture
import frame_draw

#-------------------------------
# default settings
#-------------------------------

# camera values
camera_id = 0
camera_width = 1920
camera_height = 1080
camera_frame_rate = 30
camera_fourcc = cv2.VideoWriter_fourcc(*"MJPG")

# auto measure settings
auto_percent = 0.2 
auto_threshold = 127
auto_blur = 5

# normalization settings
norm_alpha = 0
norm_beta = 255

# counting variables
object_count = 0

# history settings
history_file = 'camruler_history.json'
max_history_entries = 100

# capture settings
capture_folder = 'captured_objects'
capture_format = 'jpg'
capture_quality = 95  # for JPG (0-100)

#-------------------------------
# measurement history class
#-------------------------------

class MeasurementHistory:
    def __init__(self, filename=history_file, max_entries=max_history_entries):
        self.filename = filename
        self.max_entries = max_entries
        self.measurements = []
        self.load_history()

    def load_history(self):
        """Load measurement history from file"""
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r', encoding='utf-8') as f:
                    self.measurements = json.load(f)
            except:
                self.measurements = []
    
    def save_history(self):
        """Save measurement history to file"""
        try:
            with open(self.filename, 'w', encoding='utf-8') as f:
                json.dump(self.measurements, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving history: {e}")

    def add_measurement(self, measurement_type, x_len, y_len, l_len, area, avg_len=None, auto_mode=False):
        """Add a new measurement to history"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        measurement = {
            'timestamp': timestamp,
            'type': measurement_type,
            'x_length': round(x_len, 2),
            'y_length': round(y_len, 2),
            'diagonal_length': round(l_len, 2),
            'area': round(area, 2),
            'unit': unit_suffix,
            'auto_mode': auto_mode
        }

        if avg_len:
            measurement['average_length'] = round(avg_len, 2)
        self.measurements.append(measurement)

        # Keep only the last max_entries measurements
        if len(self.measurements) > self.max_entries:
            self.measurements = self.measurements[-self.max_entries:]
        
        self.save_history()
        print(f"HISTORY: Saved measurement - X:{x_len:.2f}, Y:{y_len:.2f}, L:{l_len:.2f}, Area:{area:.2f}")
    
    def get_recent_measurements(self, count=5):
        """Get recent measurements"""
        return self.measurements[-count:] if len(self.measurements) >= count else self.measurements
    
    def clear_history(self):
        """Clear all measurements"""
        self.measurements = []
        self.save_history()
        print("HISTORY: cleared all measurements")

    def export_to_csv(self, filename=None):
        """Export history to CSV file"""
        if not filename:
            filename = f"camruler_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        try:
            with open(filename, 'w', encoding='utf-8') as f:
                # Write header
                f.write("Timestamp,Type,X_Length,Y_Length,Diagonal_Length,Area,Average_Length,Unit,Auto_Mode\n")
                # Write data
                for m in self.measurements:
                    avg_len = m.get('average_length', '')
                    f.write(f"{m['timestamp']},{m['type']},{m['x_length']},{m['y_length']},{m['diagonal_length']},{m['area']},{avg_len},{m['unit']},{m['auto_mode']}\n")

            print(f"HISTORY: Exported to {filename}")
            return True
        except Exception as e:
            print(f"Error exporting history: {e}")
            return False

#-------------------------------
# read config file
#-------------------------------

configfile = 'camruler_config.csv'
if os.path.isfile(configfile):
    with open(configfile) as f:
        for line in f:
            line = line.strip()
            if line and line[0] != '#' and (',' in line or '=' in line):
                if ',' in line:
                    item,value = [x.strip() for x in line.split(',',1)]
                elif '=' in line:
                    item,value = [x.strip() for x in line.split('=',1)]
                else:
                    continue                        
                if item in 'camera_id camera_width camera_height camera_frame_rate camera_fourcc auto_percent auto_threshold auto_blur norm_alpha norm_beta'.split():
                    try:
                        exec(f'{item}={value}')
                        print('CONFIG:',(item,value))
                    except:
                        print('CONFIG ERROR:',(item,value))

#-------------------------------
# camera setup
#-------------------------------

if len(sys.argv) > 1:
    camera_id = sys.argv[1]
    if camera_id.isdigit():
        camera_id = int(camera_id)

camera = frame_capture.Camera_Thread()
camera.camera_source = camera_id
camera.camera_width = camera_width
camera.camera_height = camera_height
camera.camera_frame_rate = camera_frame_rate
camera.camera_fourcc = camera_fourcc
camera.start()

width = camera.camera_width
height = camera.camera_height
area = width*height
cx = int(width/2)
cy = int(height/2)
dm = hypot(cx,cy)
frate = camera.camera_frame_rate
print('CAMERA:',[camera.camera_source,width,height,area,frate])

#-------------------------------
# frame drawing/text module 
#-------------------------------

draw = frame_draw.DRAW()
draw.width = width
draw.height = height

#-------------------------------
# measurement history initialization
#-------------------------------

history = MeasurementHistory(history_file, max_history_entries)

#-------------------------------
# conversion (pixels to measure)
#-------------------------------

unit_suffix = 'mm'
pixel_base = 10
cal_range = 72
cal = dict([(x,cal_range/dm) for x in range(0,int(dm)+1,pixel_base)])
cal_base = 5
cal_last = None

def cal_update(x,y,unit_distance):
    pixel_distance = hypot(x,y)
    scale = abs(unit_distance/pixel_distance)
    target = baseround(abs(pixel_distance),pixel_base)

    low = target*scale - (cal_base/2)
    high = target*scale + (cal_base/2)

    start = target
    if unit_distance <= cal_base:
        start = 0
    else:
        while start*scale > low:
            start -= pixel_base

    stop = target
    if unit_distance >= baseround(cal_range,pixel_base):
        high = max(cal.keys())
    else:
        while stop*scale < high:
            stop += pixel_base

    for x in range(start,stop+1,pixel_base):
        cal[x] = scale
        print(f'CAL: {x} {scale}')

calfile = 'camruler_cal.csv'
if os.path.isfile(calfile):
    with open(calfile) as f:
        for line in f:
            line = line.strip()
            if line and line[0] in ('d',):
                axis,pixels,scale = [_.strip() for _ in line.split(',',2)]
                if axis == 'd':
                    print(f'LOAD: {pixels} {scale}')
                    cal[int(pixels)] = float(scale)

def conv(x,y):
    d = distance(0,0,x,y)
    scale = cal[baseround(d,pixel_base)]
    return x*scale,y*scale

def baseround(x,base=1):
    return int(base * round(float(x)/base))

def distance(x1,y1,x2,y2):
    return hypot(x1-x2,y1-y2)

#-------------------------------
# define frames
#-------------------------------

framename = "CamRuler + Object Counting + Shape Detection"
cv2.namedWindow(framename,flags=cv2.WINDOW_NORMAL|cv2.WINDOW_GUI_NORMAL)
cv2.setWindowProperty(framename, cv2.WND_PROP_TOPMOST, 1)

#-------------------------------
# key events
#-------------------------------

key_last = 0
key_flags = {'config':False,   # c key
             'auto':False,     # a key (measure dimensions)
             'count':False,    # o key (count objects)
             'thresh':False,   # t key
             'percent':False,  # p key
             'norms':False,    # n key
             'rotate':False,   # r key
             'lock':False,     # 
             'history':False,  # h key
             'hshape':False   # s key (shape classification)
             }

def key_flags_clear():
    global key_flags
    for key in list(key_flags.keys()):
        if key not in ('rotate',):
            key_flags[key] = False

def key_event(key):
    
    global key_last, key_flags, mouse_mark, cal_last

    # config mode
    if key == 99:  # 'c'
        if key_flags['config']:
            key_flags['config'] = False
        else:
            key_flags_clear()
            key_flags['config'] = True
            cal_last,mouse_mark = 0,None
    
    # history mode
    elif key == 104:  # 'h'
        key_flags['history'] = not key_flags['history']
        if key_flags['history']:
            key_flags_clear()
            key_flags['history'] = True
            mouse_mark = None

    # shape classification mode
    elif key == 115:  # 's'
        key_flags['hshape'] = not key_flags['hshape']
        if key_flags['hshape']:
            key_flags_clear()
            key_flags['hshape'] = True
            mouse_mark = None

    # normalization mode
    elif key == 110:  # 'n'
        key_flags['norms'] = not key_flags['norms']
        if key_flags['norms']:
            key_flags_clear()
            key_flags['norms'] = True
            mouse_mark = None

    # rotate
    elif key == 114:  # 'r'
        key_flags['rotate'] = not key_flags['rotate']

    # auto measure mode (A key)
    elif key == 97:  # 'a'
        key_flags['auto'] = not key_flags['auto']
        if key_flags['auto']:
            key_flags_clear()
            key_flags['auto'] = True
            mouse_mark = None

    # object counting mode (O key)
    elif key == 111:  # 'o'
        key_flags['count'] = not key_flags['count']
        if key_flags['count']:
            key_flags_clear()
            key_flags['count'] = True
            mouse_mark = None

    # auto percent
    elif key == 112 and (key_flags['auto'] or key_flags['count']):  # 'p'
        key_flags['percent'] = not key_flags['percent']
        key_flags['thresh'] = False
        key_flags['lock'] = False

    # auto threshold
    elif key == 116 and (key_flags['auto'] or key_flags['count']):  # 't'
        key_flags['thresh'] = not key_flags['thresh']
        key_flags['percent'] = False
        key_flags['lock'] = False

    # clear history (x key)
    elif key == 120:  # 'x'
        history.clear_history()

    # export history (e key)
    elif key == 101:  # 'e'
        history.export_to_csv()

    # capture object (space key)
    elif key == 32:  # space bar
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(capture_folder, f"object_{timestamp}.{capture_format}")
        
        # Create folder if not exists
        if not os.path.exists(capture_folder):
            os.makedirs(capture_folder)
        
        # Get current frame
        frame_copy = frame0.copy()
        
        # If in measurement mode, draw info on captured image
        if mouse_mark and not key_flags['auto'] and not key_flags['count']:
            x1, y1 = mouse_mark
            x2, y2 = mouse_now
            x1 += cx
            x2 += cx
            y1 *= -1
            y2 *= -1
            y1 += cy
            y2 += cy
            
            # Draw measurement info
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.line(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add text info
            x1c, y1c = conv(x1 - cx, (y1 - cy) * -1)
            x2c, y2c = conv(x2 - cx, (y2 - cy) * -1)
            xlen = abs(x1c - x2c)
            ylen = abs(y1c - y2c)
            
            cv2.putText(frame_copy, f"X: {xlen:.2f}{unit_suffix}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame_copy, f"Y: {ylen:.2f}{unit_suffix}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame_copy, f"Time: {timestamp}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Save image
        if capture_format.lower() in ('jpg', 'jpeg'):
            cv2.imwrite(filename, frame_copy, [int(cv2.IMWRITE_JPEG_QUALITY), capture_quality])
        else:
            cv2.imwrite(filename, frame_copy)
        
        print(f"CAPTURE: Saved object image to {filename}")

    print('key:',[key,chr(key) if 32 <= key <= 126 else ''])
    key_last = key

#-------------------------------
# mouse events
#-------------------------------

mouse_raw = (0,0)
mouse_now = (0,0)
mouse_mark = None

def mouse_event(event,x,y,flags,parameters):
    global mouse_raw, mouse_now, mouse_mark, key_last
    global auto_percent, auto_threshold, auto_blur, norm_alpha, norm_beta

    if key_flags['percent']:
        auto_percent = 5*(x/width)*(y/height)
    elif key_flags['thresh']:
        auto_threshold = int(255*x/width)
        auto_blur = int(20*y/height) | 1
    elif key_flags['norms']:
        norm_alpha = int(64*x/width)
        norm_beta = min(255,int(128+(128*y/height)))

    mouse_raw = (x,y)
    ox = x - cx
    oy = (y-cy)*-1
    mouse_raw = (x,y)
    
    if not key_flags['lock']:
        mouse_now = (ox,oy)

    if event == 1:  # Left click
        if key_flags['config']:
            key_flags['lock'] = False
            mouse_mark = (ox,oy)
        elif key_flags['auto'] or key_flags['count']:
            key_flags['lock'] = False
            mouse_mark = (ox,oy)
        elif key_flags['percent']:
            key_flags['percent'] = False
            mouse_mark = (ox,oy)
        elif key_flags['thresh']:
            key_flags['thresh'] = False
            mouse_mark = (ox,oy)
        elif key_flags['norms']:
            key_flags['norms'] = False
            mouse_mark = (ox,oy)
        elif not key_flags['lock']:
            if mouse_mark:
                key_flags['lock'] = True
                # Save measurement to history when locking
                x1, y1 = mouse_mark
                x2, y2 = mouse_now
                x1c, y1c = conv(x1, y1)
                x2c, y2c = conv(x2, y2)
                xlen = abs(x1c-x2c)
                ylen = abs(y1c-y2c)
                llen = hypot(xlen, ylen)
                carea = xlen*ylen
                alen = None
                if max(xlen, ylen) > 0 and min(xlen, ylen)/max(xlen, ylen) >= 0.95:
                    alen = (xlen+ylen)/2
                history.add_measurement('manual', xlen, ylen, llen, carea, alen, False)
            else:
                mouse_mark = (ox,oy)
        else:
            key_flags['lock'] = False
            mouse_now = (ox,oy)
            mouse_mark = (ox,oy)
        key_last = 0

    elif event == 2:  # Right click
        key_flags_clear()
        mouse_mark = None
        key_last = 0

cv2.setMouseCallback(framename,mouse_event)

#-------------------------------
# object counting function
#-------------------------------

def count_objects(frame):
    global object_count

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(frame_gray, (auto_blur, auto_blur), 0)
    _, thresh = cv2.threshold(blurred, auto_threshold, 255, cv2.THRESH_BINARY)
    thresh = ~thresh  # Invert binary image
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    object_count = 0  # Reset counter each frame

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        percent = 100 * w * h / area

        # Filter objects by size
        if percent < auto_percent or percent > 60:
            continue

        object_count += 1
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(frame, f'{object_count}', (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return frame, object_count

#-------------------------------
# auto measurement function
#-------------------------------

def auto_measure(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.GaussianBlur(frame_gray, (auto_blur, auto_blur), 0)
    frame_gray = cv2.threshold(frame_gray, auto_threshold, 255, cv2.THRESH_BINARY)[1]
    frame_gray = ~frame_gray
    contours, _ = cv2.findContours(frame_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    draw.crosshairs(frame, 5, weight=2, color='green')    
    
    for c in contours:
        x1, y1, w, h = cv2.boundingRect(c)
        x2, y2 = x1 + w, y1 + h
        x3, y3 = x1 + (w / 2), y1 + (h / 2)
        percent = 100 * w * h / area
        
        if percent < auto_percent or percent > 60:
            continue

        x1c, y1c = conv(x1 - cx, y1 - cy)
        x2c, y2c = conv(x2 - cx, y2 - cy)
        xlen = abs(x1c - x2c)
        ylen = abs(y1c - y2c)
        alen = 0
        if max(xlen, ylen) > 0 and min(xlen, ylen) / max(xlen, ylen) >= 0.95:
            alen = (xlen + ylen) / 2              
        carea = xlen * ylen

        draw.rect(frame, x1, y1, x2, y2, weight=2, color='red')
        draw.add_text(frame, f'{xlen:.2f}', x1 - ((x1 - x2) / 2), min(y1, y2) - 8, center=True, color='red')
        draw.add_text(frame, f'Area: {carea:.2f}', x3, y2 + 8, center=True, top=True, color='red')
        if alen:
            draw.add_text(frame, f'Avg: {alen:.2f}', x3, y2 + 34, center=True, top=True, color='green')
        if x1 < width - x2:
            draw.add_text(frame, f'{ylen:.2f}', x2 + 4, (y1 + y2) / 2, middle=True, color='red')
        else:
            draw.add_text(frame, f'{ylen:.2f}', x1 - 4, (y1 + y2) / 2, middle=True, right=True, color='red')

        # Auto save measurement for valid objects
        if percent >= auto_percent and percent <= 60:
            time.sleep(0.1)
            llen = hypot(xlen, ylen)
            history.add_measurement('auto', xlen, ylen, llen, carea, alen if alen else None, True)

    return frame

#-------------------------------
# shape classification function
#-------------------------------

def classify_shapes(frame):
    # Convert to grayscale and threshold
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    thresh = ~thresh  # invert
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        # Approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        
        # Get bounding rect
        x, y, w, h = cv2.boundingRect(approx)
        cx = x + w//2
        cy = y + h//2
        
        # Skip small contours
        if w * h < 100:  # minimum area threshold
            continue
            
        # Classify shape based on number of vertices
        shape = "unknown"
        if len(approx) == 3:
            shape = "triangle"
        elif len(approx) == 4:
            # Compute aspect ratio to distinguish square from rectangle
            aspect_ratio = float(w) / h
            shape = "square" if 0.95 <= aspect_ratio <= 1.05 else "rectangle"
        elif len(approx) == 5:
            shape = "pentagon"
        elif len(approx) == 6:
            shape = "hexagon"
        else:
            # For circles/ovals, check circularity
            area = cv2.contourArea(c)
            circularity = 4 * np.pi * area / (peri * peri)
            shape = "circle" if circularity > 0.8 else "oval"
        
        # Draw the shape and label
        draw.rect(frame, x, y, x+w, y+h, weight=1, color='blue')
        draw.add_text(frame, shape, cx, cy, color='yellow', center=True)
    
    return frame

#-------------------------------
# main loop
#-------------------------------

# Create capture folder if not exists
if not os.path.exists(capture_folder):
    os.makedirs(capture_folder)

while True:
    frame0 = camera.next(wait=1)
    if frame0 is None:
        time.sleep(0.1)
        continue

    cv2.normalize(frame0, frame0, norm_alpha, norm_beta, cv2.NORM_MINMAX)

    if key_flags['rotate']:
        frame0 = cv2.rotate(frame0, cv2.ROTATE_180)

    text = []
    fps = camera.current_frame_rate
    text.append(f'CAMERA: {camera_id} {width}x{height} {fps:.2f}FPS')

    text.append('')
    if not mouse_mark:
        text.append(f'LAST CLICK: NONE')
    else:
        text.append(f'LAST CLICK: {mouse_mark} PIXELS')
    text.append(f'CURRENT XY: {mouse_now} PIXELS')

    # History mode
    if key_flags['history']:
        draw.crosshairs(frame0, 5, weight=2, color='yellow')
        text.append('')
        text.append(f'HISTORY MODE')
        text.append(f'TOTAL MEASUREMENTS: {len(history.measurements)}')
        text.append('')

        recent = history.get_recent_measurements(5)
        if recent:
            text.append('RECENT MEASUREMENTS:')
            for i, m in enumerate(reversed(recent)):
                type_str = 'AUTO' if m['auto_mode'] else 'MANUAL'
                avg_str = f", Avg:{m.get('average_length', 'N/A')}" if m.get('average_length') else ""
                text.append(f'{i+1}. {m["timestamp"]} [{type_str}]')
                text.append(f'   X:{m["x_length"]} Y:{m["y_length"]} L:{m["diagonal_length"]}{avg_str}')
                text.append(f'   Area:{m["area"]} {m["unit"]}')
                if i < len(recent) - 1:
                    text.append('')
        else:
            text.append('NO MEASUREMENTS RECORDED')

    # Normalize mode
    elif key_flags['norms']:
        text.append('')
        text.append(f'NORMALIZE MODE')
        text.append(f'ALPHA (min): {norm_alpha}')
        text.append(f'BETA (max): {norm_beta}')
        
    # Config mode
    elif key_flags['config']:
        draw.crosshairs(frame0, 5, weight=2, color='red', invert=True)
        draw.line(frame0, cx, cy, cx + cx, cy + cy, weight=1, color='red')
        draw.line(frame0, cx, cy, cx + cy, cy - cx, weight=1, color='red')
        draw.line(frame0, cx, cy, -cx + cx, -cy + cy, weight=1, color='red')
        draw.line(frame0, cx, cy, cx - cy, cy + cx, weight=1, color='red')

        mx, my = mouse_raw
        draw.line(frame0, mx, my, mx + dm, my + (dm * (cy / cx)), weight=1, color='green')
        draw.line(frame0, mx, my, mx - dm, my - (dm * (cy / cx)), weight=1, color='green')
        draw.line(frame0, mx, my, mx + dm, my + (dm * (-cx / cy)), weight=1, color='green')
        draw.line(frame0, mx, my, mx - dm, my - (dm * (-cx / cy)), weight=1, color='green')
    
        text.append('')
        text.append(f'CONFIG MODE')

        if not cal_last:
            cal_last = cal_base
            caltext = f'CONFIG: Click on D = {cal_last}'
        elif cal_last <= cal_range:
            if mouse_mark:
                cal_update(*mouse_mark, cal_last)
                cal_last += cal_base
            caltext = f'CONFIG: Click on D = {cal_last}'
        else:
            key_flags_clear()
            cal_last == None
            with open(calfile, 'w') as f:
                data = list(cal.items())
                data.sort()
                for key, value in data:
                    f.write(f'd,{key},{value}\n')
                f.close()
            caltext = f'CONFIG: Complete.'

        draw.add_text(frame0, caltext, (cx) + 100, (cy) + 30, color='red')
        mouse_mark = None     

    # Object Counting Mode (O key)
    elif key_flags['count']:
        frame0, obj_count = count_objects(frame0)
        text.append('')
        text.append(f'OBJECT COUNTING MODE')
        text.append(f'OBJECTS DETECTED: {obj_count}')
        text.append(f'MIN SIZE: {auto_percent:.2f}%')
        draw.crosshairs(frame0, 5, weight=2, color='yellow')

    # Auto Measurement Mode (A key)
    elif key_flags['auto']:
        frame0 = auto_measure(frame0)
        text.append('')
        text.append(f'AUTO MEASURE MODE')
        text.append(f'UNITS: {unit_suffix}')
        text.append(f'MIN PERCENT: {auto_percent:.2f}')
        text.append(f'THRESHOLD: {auto_threshold}')
        text.append(f'GAUSS BLUR: {auto_blur}')

    # Shape Classification Mode (S key)
    elif key_flags['hshape']:
        frame0 = classify_shapes(frame0)
        text.append('')
        text.append(f'SHAPE CLASSIFICATION MODE')
        draw.crosshairs(frame0, 5, weight=2, color='purple')

    # Dimension mode (default)
    else:
        draw.crosshairs(frame0, 5, weight=2, color='green')    
        draw.vline(frame0, mouse_raw[0], weight=1, color='green')
        draw.hline(frame0, mouse_raw[1], weight=1, color='green')
       
        if mouse_mark:
            x1, y1 = mouse_mark
            x2, y2 = mouse_now
            x1c, y1c = conv(x1, y1)
            x2c, y2c = conv(x2, y2)
            xlen = abs(x1c - x2c)
            ylen = abs(y1c - y2c)
            llen = hypot(xlen, ylen)
            alen = 0
            if max(xlen, ylen) > 0 and min(xlen, ylen) / max(xlen, ylen) >= 0.95:
                alen = (xlen + ylen) / 2              
            carea = xlen * ylen

            text.append('')
            text.append(f'X LEN: {xlen:.2f}{unit_suffix}')
            text.append(f'Y LEN: {ylen:.2f}{unit_suffix}')
            text.append(f'L LEN: {llen:.2f}{unit_suffix}')

            x1 += cx
            x2 += cx
            y1 *= -1
            y2 *= -1
            y1 += cy
            y2 += cy
            x3 = x1 + ((x2 - x1) / 2)
            y3 = max(y1, y2)

            weight = 1
            if key_flags['lock']:
                weight = 2

            draw.rect(frame0, x1, y1, x2, y2, weight=weight, color='red')
            draw.line(frame0, x1, y1, x2, y2, weight=weight, color='green')
            draw.add_text(frame0, f'{xlen:.2f}', x1 - ((x1 - x2) / 2), min(y1, y2) - 8, center=True, color='red')
            draw.add_text(frame0, f'Area: {carea:.2f}', x3, y3 + 8, center=True, top=True, color='red')
            if alen:
                draw.add_text(frame0, f'Avg: {alen:.2f}', x3, y3 + 34, center=True, top=True, color='green')           
            if x2 <= x1:
                draw.add_text(frame0, f'{ylen:.2f}', x1 + 4, (y1 + y2) / 2, middle=True, color='red')
                draw.add_text(frame0, f'{llen:.2f}', x2 - 4, y2 - 4, right=True, color='green')
            else:
                draw.add_text(frame0, f'{ylen:.2f}', x1 - 4, (y1 + y2) / 2, middle=True, right=True, color='red')
                draw.add_text(frame0, f'{llen:.2f}', x2 + 8, y2 - 4, color='green')

    # Add usage key
    text.append('')
    text.append(f'Q = QUIT')
    text.append(f'SPACE = CAPTURE OBJECT')
    text.append(f'R = ROTATE')
    text.append(f'N = NORMALIZE')
    text.append(f'A = AUTO-MEASURE')
    text.append(f'O = OBJECT COUNTING')
    text.append(f'H = HISTORY MODE')
    text.append(f'S = SHAPE CLASSIFY')
    text.append(f'E = EXPORT HISTORY')
    text.append(f'X = CLEAR HISTORY')
    if key_flags['auto'] or key_flags['count']:
        text.append(f'P = MIN-PERCENT')
        text.append(f'T = THRESHOLD')
    text.append(f'C = CONFIG-MODE')

    draw.add_text_top_left(frame0, text)
    cv2.imshow(framename, frame0)

    key = cv2.waitKey(1) & 0xFF
    if key in (27, 113):  # ESC or Q
        break
    elif key not in (-1, 255):
        key_event(key)

#-------------------------------
# shutdown
#-------------------------------

camera.stop()
cv2.destroyAllWindows()