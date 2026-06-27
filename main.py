import cv2
import face_recognition
import time
import os

# -------------------------------------------------------------------------
# THIẾT LẬP HỆ THỐNG & TĂNG TỐC PHẦN CỨNG (GPU)
# -------------------------------------------------------------------------
IMAGE_FOLDER = "images"
MAX_DISPLAY_WIDTH = 800

# Kích hoạt tăng tốc phần cứng cho OpenCV nếu máy bạn có GPU (Nvidia/Intel/AMD)
# Điều này giúp hàm cv2.resize hoặc các hàm xử lý ma trận ảnh chạy thẳng trên GPU.
cv2.setUseOptimized(True)
if cv2.useOptimized():
    print("[HỆ THỐNG] Đã kích hoạt chế độ tối ưu hóa phần cứng đồ họa OpenCV.")

if not os.path.exists(IMAGE_FOLDER):
    print(f"Lỗi: Không tìm thấy thư mục {IMAGE_FOLDER}")
    exit()

valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')
image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(valid_extensions)]
print(f"Tìm thấy {len(image_files)} file ảnh.")

# -------------------------------------------------------------------------
# VÒNG LẶP DUYỆT ẢNH
# -------------------------------------------------------------------------
for index, filename in enumerate(image_files):
    image_path = os.path.join(IMAGE_FOLDER, filename)
    print(f"\nProcessing [{index + 1}/{len(image_files)}]: {filename}")

    image_raw = cv2.imread(image_path)
    if image_raw is None:
        continue

    # Chuẩn hóa kích thước
    h, w = image_raw.shape[:2]
    if w > MAX_DISPLAY_WIDTH:
        scale = MAX_DISPLAY_WIDTH / w
        image_display = cv2.resize(image_raw, (MAX_DISPLAY_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        image_display = image_raw.copy()

    image_hog = image_display.copy()
    image_cnn = image_display.copy()
    rgb_image = cv2.cvtColor(image_display, cv2.COLOR_BGR2RGB)

    # --- [1] XỬ LÝ HOG ---
    start_hog = time.time()
    face_locations_hog = face_recognition.face_locations(rgb_image, number_of_times_to_upsample=1, model="hog")
    print(f" -> HOG: {len(face_locations_hog)} mặt ({time.time() - start_hog:.2f}s)")

    for top, right, bottom, left in face_locations_hog:
        cv2.rectangle(image_hog, (left, top), (right, bottom), (0, 255, 0), 2)

    # --- [2] XỬ LÝ CNN ---
    start_cnn = time.time()
    # Mẹo GPU: Nếu bạn chạy trên máy có hỗ trợ phần cứng, mô hình CNN của dlib
    # sẽ tự động nhận diện. Nếu không, nó vẫn chạy trên các nhân tính toán logic của CPU.
    face_locations_cnn = face_recognition.face_locations(rgb_image, number_of_times_to_upsample=1, model="cnn")
    print(f" -> CNN: {len(face_locations_cnn)} mặt ({time.time() - start_cnn:.2f}s)")

    for top, right, bottom, left in face_locations_cnn:
        cv2.rectangle(image_cnn, (left, top), (right, bottom), (0, 0, 255), 2)

    # --- KHỞI TẠO CỬA SỔ HIỂN THỊ ---
    win_hog = "Phuong phap HOG"
    win_cnn = "Phuong phap CNN"

    cv2.namedWindow(win_hog, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_cnn, cv2.WINDOW_NORMAL)

    cv2.imshow(win_hog, image_hog)
    cv2.imshow(win_cnn, image_cnn)

    # --- LUỒNG BẮT SỰ KIỆN BÀN PHÍM VÀ CHUỘT (FIX LỖI DẤU X) ---
    print("[HƯỚNG DẪN]: Bấm phím bất kỳ để chuyển ảnh. Bấm 'q' để thoát.")

    while True:
        # Kiểm tra xem người dùng có bấm nút X tắt cửa sổ thủ công bằng chuột không
        # Nếu giá trị trả về < 0 nghĩa là cửa sổ đã bị đóng bởi nút X bên ngoài
        prop_hog = cv2.getWindowProperty(win_hog, cv2.WND_PROP_VISIBLE)
        prop_cnn = cv2.getWindowProperty(win_cnn, cv2.WND_PROP_VISIBLE)

        if prop_hog < 1 or prop_cnn < 1:
            print(" -> Phát hiện người dùng đóng cửa sổ bằng nút X.")
            key = ord('n')  # Tự động gán lệnh Next ảnh
            break

        # Đợi phím bấm từ bàn phím trong vòng 100ms (Không treo luồng để vòng lặp kiểm tra nút X liên tục chạy)
        key = cv2.waitKey(100) & 0xFF
        if key != 255:  # Nếu có phím thực tế được bấm
            break

    # GIẢI PHÓNG CỬA SỔ AN TOÀN: Chỉ đóng những cửa sổ nào còn tồn tại
    if cv2.getWindowProperty(win_hog, cv2.WND_PROP_VISIBLE) >= 1:
        cv2.destroyWindow(win_hog)
    if cv2.getWindowProperty(win_cnn, cv2.WND_PROP_VISIBLE) >= 1:
        cv2.destroyWindow(win_cnn)

    if key == ord('q'):
        print("\nĐã dừng tiến trình theo yêu cầu.")
        break

cv2.destroyAllWindows()
print("\n[HOÀN THÀNH] Hoàn tất tiến trình quét dữ liệu.")