# ĐỒ ÁN CUỐI KỲ: NHẬP MÔN XỬ LÝ ẢNH SỐ 
**Đề tài:** Ứng dụng xử lý ảnh để đo kích thước vật thể trong ảnh (đo chiều dài, diện tích)

---

Chương trình là một tập lệnh Python sử dụng OpenCV để ghi lại video từ camera và đo kích thước của các đối tượng trong luồng video. Nó cũng đếm số lượng đối tượng trong khung hình video.

# Các tính năng chính

* Đo kích thước của các đối tượng trong luồng video
* Đếm số lượng đối tượng trong khung hình video
* Lưu lịch sử đo lường vào tệp
* Chụp màn hình
* Nhận diện hình học

## Chức năng đếm số lượng đối tượng - phím O (OBJECT COUNTING)
### Mục đích
Đếm số lượng các vật thể tìm thấy trong khung hình
### Ý tưởng thực hiện
Khi người dùng nhấn phím O, chương trình sẽ hoạt động như sau:
1. Chuyển đổi khung hình sang ảnh xám
2. Làm mờ ảnh để giảm nhiễu
3. Ngưỡng hóa ảnh xám thành ảnh nhị phân (đen/trắng)
4. Đảo ngược ảnh nhị phân
5. Tìm các đường viền trong ảnh thông qua hàm cv2.findContours
6. Khởi tạo biến đếm đối tượng
7. Duyệt qua từng vật thể được tìm thấy
8. Bỏ qua các vật thể quá nhỏ hoặc quá lớn (nhỏ hơn 0.2, lớn hơn 60 % khung hình)
### Công thức đã sử dụng
Tính diện tích vật thể so với diện tích khung hình
```python
percent = 100 * w * h / area
```
> w = chiều rộng | h = chiều cao | area = khung hình
### Ví dụ hoạt động
Khi người dùng nhấn phím O chương trình sẽ chạy hàm count_objects để tìm kiếm các đối tượng trong khung hình
### Code chính
```python
def count_objects(frame):
    global object_count

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    # chuyển đổi khung hình sang ảnh xám
    blurred = cv2.GaussianBlur(frame_gray, (auto_blur, auto_blur), 0) # làm mờ ảnh để giảm nhiễu
    _, thresh = cv2.threshold(blurred, auto_threshold, 255, cv2.THRESH_BINARY) # ngưỡng hóa ảnh xám thành ảnh nhị phân (đen/trắng)
    thresh = ~thresh  # đảo ngược ảnh nhị phân 
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # tìm các đường viền trong ảnh
    object_count = 0  # khởi tạo biến đếm đối tượng

    for c in contours:  # duyệt qua từng vật thể được tìm thấy
        x, y, w, h = cv2.boundingRect(c) #tính hcn bao quanh vật thể
        percent = 100 * w * h / area #tính diện tích vật thể so với diện tích khung hình

        # bỏ qua các vật thể quá nhỏ hoặc quá lớn
        if percent < auto_percent or percent > 60:
            continue

        object_count += 1
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(frame, f'{object_count}', (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return frame, object_count
```

# Chức năng lưu lịch sử đo lường (HISTORY SAVE)

### Mục đích

Lưu lại lịch sử các thao tác hoặc kết quả xử lý (ví dụ: số lượng đối tượng đếm được, thời gian thao tác, trạng thái khung hình) để người dùng có thể xem lại hoặc xuất ra file.

### Thư viện đã sử dụng

```py
import json #lưu lịch sử đo dưới dạng .json
import csv # lưu lịch sử đo dưới dạng .csv
from datetime import datetime # xác định thời gian thực hiện phép đo
import os # kiểm tra và tạo thư mục nếu chưa tồn tại
```

### Ý tưởng thực hiện

Khi người dùng nhấn phím H, chương trình sẽ hoạt động như sau:

1. Tạo lớp `MeasurementHistory` để quản lý lịch sử đo lường.
2. Lịch sử sẽ được lưu trong file `camruler_history.json`.
3. Mỗi lần đo, một đối tượng `dict` được tạo chứa các thông tin:

- Loại đo (`auto`, `manual`)
- Kích thước đo được (chiều dài x, y, đường chéo, diện tích)
- Thời gian đo
- Ghi chú (nếu cần)

4. Có thể xuất lịch sử ra định dạng `.csv` để dễ phân tích.
5. Có thể xóa toàn bộ lịch sử đo lường khi nhấn phím `e`.

### Công thức đã sử dụng

- Chiều dài cạnh:

```py
llen = hypot(xlen, ylen)
```

- Chiều dài trung bình (nếu hình vuông hoặc gần vuông):

```py
avg_len = (x_length + y_length) / 2
```

- Lưu thời gian hiện tại:

```py
from datetime import datetime
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
```

### Ví dụ hoạt động

Sau mỗi lần đếm đối tượng và đo vật thể, khi nhấn phím H, chương trình sẽ lưu lại số lượng đối tượng và thời gian vào danh sách `history`. Khi cần, có thể xuất danh sách này ra file CSV.

Sau khi thực hiện phép đo, kết quả sẽ được lưu như sau:

```json
{
  "timestamp": "2025-07-25 17:45:21",
  "type": "manual",
  "x_length": 32.5,
  "y_length": 48.9,
  "diagonal_length": 58.3,
  "area": 1602.7,
  "avg_len": null,
  "unit": 100,
  "auto_mode": false
}
```

Bạn có thể nhấn `h` để xem lịch sử đo trên giao diện và nhấn `e` để xuất toàn bộ dữ liệu đo thành file `camruler_export.csv`

Bạn cũng có thể xóa toàn bộ lịch sử đo lường bằng cách nhấn `e` để xóa.

### Code chính

```py
class MeasurementHistory:
    def __init__(self, filename=history_file, max_entries=max_history_entries):
        self.filename = filename
        self.max_entries = max_entries
        self.measurements = []
        self.load_history()

    def load_history(self):
        """tải lịch sử đo lường từ file"""
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r', encoding='utf-8') as f:
                    self.measurements = json.load(f)
            except:
                self.measurements = []

    def save_history(self):
        """Lưu lịch sử đo lường vào file"""
        try:
            with open(self.filename, 'w', encoding='utf-8') as f:
                json.dump(self.measurements, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving history: {e}")

    def add_measurement(self, measurement_type, x_len, y_len, l_len, area, avg_len=None, auto_mode=False):
        """Thêm một đo lường mới vào lịch sử"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        measurement = {
            'timestamp': timestamp,
            'type': measurement_type, # 'tự động' hoặc 'thủ công'
            'x_length': round(x_len, 2), # chiều dài trục x
            'y_length': round(y_len, 2), # chiều dài trục y
            'diagonal_length': round(l_len, 2), # chiều dài đường chéo
            'area': round(area, 2), # diện tích hình chữ nhật
            'unit': unit_suffix, # đơn vị đo lường (vd: mm)
            'auto_mode': auto_mode
        }
    def clear_history(self):
        """Xóa tất cả lịch sử đo lường"""
        self.measurements = []
        self.save_history()
        print("HISTORY: cleared all measurements")

    def export_to_csv(self, filename=None):
        """Xuất lịch sử vào file CSV"""
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
```


## Chức năng chụp màn hình - Object capture ( phím space )
### Mục Đích
- Cho phép người dùng lưu lại hình ảnh đang hiển thị từ camera bằng cách nhấn phím Space (phím cách).
- Nếu đang ở chế độ đo đạc, ảnh sẽ được ghi lại kèm theo thông tin đo chiều dài/rộng và thời gian.
- Giúp lưu trữ dữ liệu dưới dạng ảnh để phân tích, báo cáo hoặc kiểm tra sau.
### Ý tưởng thực hiện 
Khi người dùng nhấn phím cách, chương trình sẽ hoạt động như sau:
1. Ghi lại thời điểm chụp lúc đó để làm tên file
2. Lưu ảnh gốc ( frame).
3. Nếu đang sử dụng các tính năng vào vật thể, lấy luôn các khuôn, số liệu, và các thông tin vật thể vào ảnh
4. Ghi ảnh ra ổ đĩa với định dạng và chất lượng đã thiết lập
### Công thức xử lí đo đạc
- (x1, y1): điểm bắt đầu khi rê chuột.
- (x2, y2): điểm hiện tại con trỏ chuột.
- cx, cy: tọa độ tâm ảnh (để quy đổi về hệ tọa độ chuẩn).
- conv(): hàm chuyển đổi tọa độ pixel sang đơn vị đo thực tế (ví dụ: mm hoặc cm).
- Quy đổi:
```python
x1' = x1 + cx
x2' = x2 + cx
y1' = -y1 + cy
y2' = -y2 + cy
```
### Ví dụ hoạt động 
1. Khi người dùng đo một vật thể có chiều ngang và dọc, rồi nhấn phím cách (space)
2. Một ảnh sẽ được lưu vào folder capture_objects, tên của ảnh giống như thế này vd: object_20250722_002131.jpg
3. Trong ảnh có thể chỉ là hình chụp vật thể hoặc đi kèm với các tính năng đã sử dụng như khung chữ nhật đánh dấu vật thể, số liệu đo, chiều dài tính theo đơn vị,...
### Code chính
```python
elif key == 32:  # space bar
    # Tạo tên file với timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(capture_folder, f"object_{timestamp}.{capture_format}")
    
    # Tạo thư mục nếu chưa có
    if not os.path.exists(capture_folder):
        os.makedirs(capture_folder)
    
    # Sao chép frame hiện tại
    frame_copy = frame0.copy()
    
    # Nếu đang đo và không ở chế độ tự động hay đếm
    if mouse_mark and not key_flags['auto'] and not key_flags['count']:
        x1, y1 = mouse_mark
        x2, y2 = mouse_now
        
        # Quy đổi tọa độ chuột sang tọa độ thực tế trên ảnh
        x1 += cx; x2 += cx
        y1 = -y1 + cy
        y2 = -y2 + cy
        
        # Vẽ hình chữ nhật đo đạc và đường nối
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.line(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Tính chiều dài X và Y sau khi chuyển đổi đơn vị
        x1c, y1c = conv(x1 - cx, (y1 - cy) * -1)
        x2c, y2c = conv(x2 - cx, (y2 - cy) * -1)
        xlen = abs(x1c - x2c)
        ylen = abs(y1c - y2c)
        
        # Hiển thị thông tin lên ảnh
        cv2.putText(frame_copy, f"X: {xlen:.2f}{unit_suffix}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame_copy, f"Y: {ylen:.2f}{unit_suffix}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame_copy, f"Time: {timestamp}", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Ghi ảnh ra file theo định dạng và chất lượng đã chọn
    if capture_format.lower() in ('jpg', 'jpeg'):
        cv2.imwrite(filename, frame_copy, [int(cv2.IMWRITE_JPEG_QUALITY), capture_quality])
    else:
        cv2.imwrite(filename, frame_copy)
    
    print(f"CAPTURE: Saved object image to {filename}")
```

## Chức năng nhận diện hình học - shape detection ( phím s )
### Mục đích
- Tự động phân loại các hình cơ bản như: tam giác, hình vuông, chữ nhật, hình tròn,...
- Hữu ích trong việc ứng dụng đo lường, phân loại vật thể, nhận dạng mẫu,...
### Ý tưởng thực hiện
- Dùng xử lí ảnh để tìm biên đối tượng
- Dựa vào số lượng đỉnh (vertices) và tỷ lệ cạnh, xác định hình dạng.
- Đối Với hình tròn, kiểm tra độ tròn (circularity) dựa vào diện tích và chu vi.
### Công thức toán học đã xử dụng
- Chu vi (Perimeter) của đường bao:
```python
P = cv2.arcLength(contour, True)
```
- Gần đúng đa giác (Polygon approximation):
```python
approx = cv2.approxPolyDP(contour, 0.04 * P, True)
```
- Tỷ lệ khung chữ nhật (Aspect Ratio) – phân biệt vuông và chữ nhật:
```python
aspect_ratio = width / height
```
- Độ tròn (Circularity) – dùng để phân biệt hình tròn:
```python
circularity = 4 * π * area / (perimeter)^2 
```
- Nếu tính ra > 0.8 thì là hình tròn, nhỏ hơn thì là elip hoặc hình khác
### Ví dụ hoạt động
> Vật có 3 đỉnh gắn label là triangle, 4 thì là square, tròn là circle,...
# code chính
```python 
def classify_shapes(frame):
    # Chuyển ảnh sang xám và làm mờ nhẹ
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Nhị phân ảnh và đảo ngược màu
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    thresh = ~thresh  # đảo màu trắng đen
    
    # Tìm đường bao (contours)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        # Bỏ qua các vùng quá nhỏ
        x, y, w, h = cv2.boundingRect(c)
        if w * h < 100:
            continue

        # Gần đúng đa giác từ contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        
        # Tâm đối tượng
        cx = x + w // 2
        cy = y + h // 2
        
        # Phân loại hình dựa theo số đỉnh
        shape = "unknown"
        if len(approx) == 3:
            shape = "triangle"
        elif len(approx) == 4:
            aspect_ratio = float(w) / h
            shape = "square" if 0.95 <= aspect_ratio <= 1.05 else "rectangle"
        elif len(approx) == 5:
            shape = "pentagon"
        elif len(approx) == 6:
            shape = "hexagon"
        else:
            area = cv2.contourArea(c)
            circularity = 4 * np.pi * area / (peri * peri)
            shape = "circle" if circularity > 0.8 else "oval"
        
        # Vẽ khung và tên hình dạng
        draw.rect(frame, x, y, x+w, y+h, weight=1, color='blue')
        draw.add_text(frame, shape, cx, cy, color='yellow', center=True)

    return frame

```
