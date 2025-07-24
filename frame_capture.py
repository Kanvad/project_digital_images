import time
import threading, queue
import numpy as np
import cv2

# ------------------------------
# Luồng xử lý camera (Camera Thread)
# ------------------------------

class Camera_Thread:

    # Lưu ý: sử dụng hàng đợi (queue) sẽ hiệu quả hơn rất nhiều so với deque.
    # Chạy bằng queue chỉ chiếm khoảng 35% CPU, trong khi deque chiếm tới 108%.

    # ------------------------------
    # Hướng dẫn sử dụng
    # ------------------------------

    # Các biến người dùng có thể tùy chỉnh bên dưới:
    # - Chọn nguồn camera (mặc định là camera 0).
    # - Chỉnh độ phân giải (mặc định 640x480).
    # - Cài đặt tốc độ khung hình tối đa (mặc định 30 fps).
    # - Cài số khung hình lưu trong bộ đệm (mặc định là 4).
    # - `buffer_all`: True = không mất khung hình (phù hợp khi đọc từ file); 
    #                 False = cho phép mất khung hình (dùng với camera thật, giữ frame mới nhất).

    # Khởi động camera bằng cách gọi `self.start()`.

    # Lấy frame tiếp theo bằng cách gọi `self.next(black=True, wait=1)`
    #    - Nếu `black=True`, giá trị trả về mặc định là khung hình đen nếu không có khung nào.
    #    - Nếu `black=False`, giá trị mặc định là `None`.
    #    - Tham số `wait`: thời gian chờ tối đa để nhận được frame.
    #    - Nếu không có frame trong buffer, trả về giá trị mặc định (đen hoặc None).

    # Dừng camera bằng cách gọi `self.stop()`

    # ------------------------------
    # User Variables
    # ------------------------------

    camera_source = 0                  # Số hiệu camera (0 là mặc định)
    camera_width = 640                # Chiều rộng video
    camera_height = 480               # Chiều cao video
    camera_frame_rate = 30            # Số khung hình/giây mong muốn
    camera_fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # Định dạng mã hóa video
    # Tuỳ chọn thay thế: camera_fourcc = cv2.VideoWriter_fourcc(*"YUYV")

    buffer_length = 5                 # Số khung hình giữ trong bộ đệm
    buffer_all = False                # Có giữ lại toàn bộ khung hình không

    # ------------------------------
    # System Variables
    # ------------------------------

    camera = None                     # Đối tượng camera OpenCV
    camera_init = 0.5                 # Thời gian chờ sau khi mở camera

    buffer = None                     # Bộ đệm khung hình

    frame_grab_run = False            # Biến điều khiển chạy luồng camera
    frame_grab_on = False             # Trạng thái đang hoạt động

    frame_count = 0                   # Tổng số khung hình đã đọc
    frames_returned = 0               # Số khung hình đã lấy ra
    current_frame_rate = 0            # Tốc độ khung hình hiện tại (fps)
    loop_start_time = 0              # Thời điểm bắt đầu vòng lặp

    # ------------------------------
    # Hàm khởi động camera và luồng xử lý
    # ------------------------------

    def start(self):
        # Khởi tạo buffer tùy theo chế độ
        if self.buffer_all:
            self.buffer = queue.Queue(self.buffer_length)
        else:
            self.buffer = queue.Queue(1)  # Chỉ giữ frame mới nhất

        # Cài đặt thông số cho camera
        self.camera = cv2.VideoCapture(self.camera_source)
        self.camera.set(3, self.camera_width)
        self.camera.set(4, self.camera_height)
        self.camera.set(5, self.camera_frame_rate)
        self.camera.set(6, self.camera_fourcc)
        time.sleep(self.camera_init)  # Chờ camera sẵn sàng

        # Cập nhật lại thông số từ camera thực tế
        self.camera_width  = int(self.camera.get(3))
        self.camera_height = int(self.camera.get(4))
        self.camera_frame_rate = int(self.camera.get(5))
        self.camera_mode = int(self.camera.get(6))
        self.camera_area = self.camera_width * self.camera_height

        # Tạo một khung hình đen (dùng làm khung mặc định)
        self.black_frame = np.zeros((self.camera_height, self.camera_width, 3), np.uint8)

        # Bắt đầu luồng lấy khung hình
        self.frame_grab_run = True
        self.thread = threading.Thread(target=self.loop)
        self.thread.start()

    # ------------------------------
    # Hàm dừng camera và giải phóng tài nguyên
    # ------------------------------
    def stop(self):
        self.frame_grab_run = False  # Yêu cầu dừng vòng lặp

        while self.frame_grab_on:    # Thời gian chờ vòng lặp dừng hẳn
            time.sleep(0.1)

        # Giải phóng camera nếu chưa dừng
        if self.camera:
            try:
                self.camera.release()
            except:
                pass
        self.camera = None
        self.buffer = None           # Xoá bộ đệm

    # ------------------------------
    # Vòng lặp chính để đọc và đẩy khung hình vào buffer
    # ------------------------------
    def loop(self):
        frame = self.black_frame

        # Thêm khung đầu tiên vào buffer
        if not self.buffer.full():
            self.buffer.put(frame, False)

        self.frame_grab_on = True
        self.loop_start_time = time.time()

        fc = 0  # Đếm số frame đọc được để tính fps
        t1 = time.time()

        while True:
            if not self.frame_grab_run:
                break  # Dừng nếu có lệnh tắt

            if self.buffer_all:
                # Nếu buffer đầy thì chờ cho trống
                if self.buffer.full():
                    time.sleep(1 / self.camera_frame_rate)
                else:
                    # Đọc frame từ camera
                    grabbed, frame = self.camera.read()
                    if not grabbed:
                        break
                    self.buffer.put(frame, False)
                    self.frame_count += 1
                    fc += 1
            else:
                # Cho phép mất frame (dùng với camera)
                grabbed, frame = self.camera.read()
                if not grabbed:
                    break

                if self.buffer.full():
                    self.buffer.get()  # Xoá frame cũ nhất

                self.buffer.put(frame, False)
                self.frame_count += 1
                fc += 1

            # Cập nhật tốc độ đọc khung hình sau mỗi 10 frame trở lên
            if fc >= 10:
                self.current_frame_rate = round(fc / (time.time() - t1), 2)
                fc = 0
                t1 = time.time()

        # Dọn dẹp khi kết thúc
        self.loop_start_time = 0
        self.frame_grab_on = False
        self.stop()

    # ------------------------------
    # Hàm lấy khung hình kế tiếp từ buffer
    # ------------------------------
    def next(self, black=True, wait=0):

        # Mặc định là khung hình đen hoặc None
        frame = self.black_frame if black else None

        # Lấy frame từ buffer nếu có
        try:
            frame = self.buffer.get(timeout=wait)
            self.frames_returned += 1
        except queue.Empty:
            # Nếu hết thời gian chờ và không có frame thì giữ frame mặc định
            pass

        return frame
