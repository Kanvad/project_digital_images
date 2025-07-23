import cv2

# ------------------------------
# Draw on Frame
# ------------------------------


class DRAW:

    # ------------------------------
    # User Variables
    # ------------------------------

    # frame
    width = 640  # Chiều rộng khung hình
    height = 480  # Chiều cao khung hình

    # Định dạng màu theo BGR (Blue, Green, Red)
    colors = {'red': (0,  0, 255),
              'green': (0, 255,  0),
              'blue': (255,  0,  0),
              'yellow': (0, 255, 255),
              'gray': (200, 200, 200),
              }

    # ------------------------------
    # Functions
    # ------------------------------

    # Hàm thêm văn bản phía trên bên trái khung hình
    def add_text_top_left(self, frame, text):

        if type(text) not in (list, tuple):
            text = text.split('\n')  # Chia thành danh sách nếu là chuỗi
        text = [line.rstrip() for line in text]

        # Màu mặc định là 'blue', nếu không có sẽ dùng màu vàng (0,255,255)
        color = self.colors.get('blue', (0, 255, 255))

        lineloc = 10  # Vị trí bắt đầu của dòng chữ theo chiều dọc
        lineheight = 30  # Chiều cao dòng chữ

        for line in text:
            lineloc += lineheight  # Vị trí dòng chữ tiếp theo
            # Vẽ dòng chữ trên khung hình
            cv2.putText(frame,  # Khung hình hiển thị
                        line,  # Nội dung văn bản
                        (10, lineloc),  # Vị trí hiển thị văn bản
                        cv2.FONT_HERSHEY_SIMPLEX,  # Kiểu font chữ
                        0.8,  # Kích thước chữ
                        color,  # Màu chữ
                        1,  # Độ dày của nét chữ
                        cv2.LINE_AA,  # Chống răng cưa cho chữ
                        False)  # Sử dụng viền (False)

    # Hàm thêm văn bản
    def add_text(self, frame, text, x, y, size=0.8, color='yellow', center=False, middle=False, top=False, right=False):

        color = self.colors.get(color, (0, 255, 255))
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Tính kích thước của văn bản để điều chỉnh vị trí
        textsize = cv2.getTextSize(text, font, size, 1)[0]

        if center:
            x -= textsize[0]/2  # Căn giữa theo chiều ngang
        elif right:
            x -= textsize[0]  # Căn phải
        if top:
            y += textsize[1]  # Căn trên
        elif middle:
            y += textsize[1]/2  # Căn giữa theo chiều dọc

        cv2.putText(frame,
                    text,
                    # x, y được ép kiểu về int để tránh lỗi khi truyền số thực
                    (int(x), int(y)),
                    font,
                    size,
                    color,
                    1,
                    cv2.LINE_AA,
                    False)

    # Hàm vẽ đường thẳng
    def line(self, frame, x1, y1, x2, y2, weight=1, color='green'):
        # Vẽ một đường thẳng trên khung hình 'frame' từ điểm (x1, y1) đến điểm (x2, y2)
        # 'weight' là độ dày của đường thẳng, mặc định là 1
        # 'color' là màu của đường thẳng, mặc định là 'green'
        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                 self.colors.get(color, (0, 255, 0)), weight)

    # Hàm vẽ đường thẳng đứng
    def vline(self, frame, x=0, weight=1, color='green'):
        # Nếu giá trị x nhỏ hơn hoặc bằng 0, đặt x bằng nửa chiều rộng của khung hình
        if x <= 0:
            x = self.width/2
        x = int(x)

        cv2.line(frame, (x, 0), (x, self.height),
                 self.colors.get(color, (0, 255, 0)), weight)

    # Hàm vẽ đường thẳng ngang
    def hline(self, frame, y=0, weight=1, color='green'):
        # Nếu giá trị y nhỏ hơn hoặc bằng 0, đặt y bằng nửa chiều cao của khung hình
        if y <= 0:
            y = self.height/2
        y = int(y)

        cv2.line(frame, (0, y), (self.width, y),
                 self.colors.get(color, (0, 255, 0)), weight)

    # Hàm vẽ hình chữ nhật
    def rect(self, frame, x1, y1, x2, y2, weight=1, color='green', filled=False):
        # Nếu tham số 'filled' là True, đặt độ dày 'weight' thành -1 để vẽ hình chữ nhật đầy đủ
        if filled:
            weight = -1
        # Vẽ hình chữ nhật trên khung hình 'frame' từ điểm (x1, y1) đến điểm (x2, y2)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                      self.colors.get(color, (0, 255, 0)), weight)

    # Hàm vẽ hình tròn
    def circle(self, frame, x1, y1, r, weight=1, color='green', filled=False):
        # Nếu tham số 'filled' là True, đặt độ dày 'weight' là -1 để vẽ hình tròn đày đủ
        if filled:
            weight = -1
        # Vẽ hình tròn trên khung hình 'frame' với tâm tại (x1, y1) và bán kính 'r'
        cv2.circle(frame, (int(x1), int(y1)),
                   int(r), self.colors.get(color, (0, 255, 0)), weight)

    # Hàm vẽ dấu chéo ở trung tâm toàn khung hình
    def crosshairs_full(self, frame, weight=1, color='green'):
        # Vẽ đường thẳng đứng ở giữa khung hình
        self.vline(frame, 0, weight, color)
        # Vẽ đường thẳng ngang ở giữa khung hình
        self.hline(frame, 0, weight, color)

    # Hàm vẽ dấu chéo trung tâm với độ lệch
    def crosshairs(self, frame, offset=10, weight=1, color='green', invert=False):
        # Tính toán độ lệch dựa trên chiều rộng của khung hình
        offset = self.width*offset/200
        # Tính toán tọa độ trung tâm của khung hình
        xcenter = self.width/2
        ycenter = self.height/2

        if invert:
            # Nếu 'invert' là True, vẽ dấu chéo ngược
            self.line(frame, 0, ycenter, xcenter -
                      offset, ycenter, weight, color)  # Vẽ đường thẳng bên trái
            self.line(frame, xcenter+offset, ycenter,
                      self.width, ycenter, weight, color)  # Vẽ đường thẳng bên phái
            self.line(frame, xcenter, 0, xcenter,
                      ycenter-offset, weight, color)  # Vẽ đường thẳng dưới
            self.line(frame, xcenter, ycenter+offset,
                      xcenter, self.height, weight, color)  # Vẽ đường thẳng trên
        else:
            # Nếu 'invert' là False, vẽ dấu chéo bình thường
            self.line(frame, xcenter-offset, ycenter,
                      xcenter+offset, ycenter, weight, color)  # Vẽ đường thẳng ngang
            self.line(frame, xcenter, ycenter-offset,
                      xcenter, ycenter+offset, weight, color)  # Vẽ đường thẳng đứng

# ------------------------------
# end
# ------------------------------
