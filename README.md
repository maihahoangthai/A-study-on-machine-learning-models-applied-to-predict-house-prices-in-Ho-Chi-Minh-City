# A study on machine learning models applied to predict house prices in Ho Chi Minh City
## Giới thiệu chương trình
Một trang web đơn giản để demo đề tài "Tìm hiểu các mô hình học máy ứng dụng vào dự đoán giá nhà tại TP.HCM năm 2023".
## Về dữ liệu
dataset.csv: tập dữ liệu thô, chứa 20.499 mẫu tin đăng bán nhà thu thập từ website batdongsan.com.vn.

datasetAfterDataCleaning.csv: dữ liệu đã qua xử lý, làm sạch từ tập dữ liệu thô ban đầu.

kv_px_KhoangGiaDict.pkl: chứa dữ liệu về 'Khoảng giá phổ biến Min', 'Giá phổ biến nhất', 'Khoảng giá phổ biến Max' theo từng cặp 'Khu vực' với 'phường/xã'.

kv_px_shdDict.pkl: chứa dữ liệu về 'dự án/số hẻm đường' theo từng cặp 'Khu vực' với 'phường/xã'.

Thư mục Encoder: các encoder để chuyển đổi dữ liệu dạng categorical sang numerical.

Thư mục Model: các mô hình học máy đã được huấn luyện để ứng dụng vào dự đoán giá nhà.

Thư mục Parameter: những tham số tốt nhất để tạo từng mô hình sau khi áp dụng Hyperparameter tuning thông qua GridSearchCV.

Thư mục Scaler: các scaler để scaling dữ liệu về mức [0, 1] dựa trên kỹ thuật Standardization.

Thư mục templates và static: Front-end của web.

app.py: Back-end của web.
## Cách chạy web demo dự đoán giá nhà trong môi trường máy tính local
Bước 1: Mở cmd hoặc terminal hoặc windows PowerShell và chuyển đường dẫn đến thư mục có chứa file 'app.py'.

Bước 2: Đã cài đặt ít nhất Python phiên bản 3.11.4 và các thư viện tiên quyết như 'scikit-learn', 'numpy', 'flask', 'pandas', 'xgboost'.

Bước 3: Gõ lệnh 'python app.py' và copy đường dẫn http hiện lên.

Bước 4: Truy cặp web bằng link localhost vừa copy.
