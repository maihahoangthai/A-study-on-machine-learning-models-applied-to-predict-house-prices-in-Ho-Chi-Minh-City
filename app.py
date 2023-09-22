# Code thay đổi thư mục làm việc:
#import os
#print("Đường dẫn của thư mục hiện tại:", os.getcwd()) # ví dụ: 'C:\Users\ADMIN'.
#os.chdir('C:/Users/ADMIN/Documents/MiniConda/DACNTT2') # Lệnh thay đổi thư mục.
#print("Đường dẫn của thư mục hậu thay đổi:", os.getcwd()) # ví dụ: 'C:/Users/ADMIN/Documents/MiniConda/DACNTT2'.

# Nếu chưa cài đặt sklearn thì gõ "pip install scikit-learn==1.2.2" vào cmd. 
# Lưu ý: nếu file pickle lưu từ Google Colab và sklearn của máy local có version khác nhau,
# thì dễ gặp tình trạng lỗi hoặc warning khi cố unpickle nó. 
# Do đó, cần downgrade hoặc upgrade sklearn về cùng phiên bản với file pickle.
# Ví dụ: "sc_X.pkl" sử dụng phiên bản 1.2.2, nhưng local đang cài sklearn bản 1.3.0. Vì vậy, cần downgrade local về 1.2.2.
#import sklearn.external.joblib as extjoblib
import numpy as np
import joblib # Nếu chưa cài đặt joblib thì gõ "pip install joblib" vào cmd.
from flask import Flask, render_template, request # Nếu chưa cài đặt flask thì gõ "pip install flask" vào cmd.
import pandas as pd
import statistics
import math
import random
from xgboost import XGBRegressor # Nếu chưa cài đặt xgboost thì gõ "pip install xgboost" vào cmd. Không cài xgboost đồng nghĩa với việc không thể unpickle model XGBoost.

# Khởi tạo một "app" object bằng Flask class:
app = Flask(__name__)

# Nạp các file pickle đã chuẩn bị từ trước:
# Giả sử thư mục hiện tại đang là 'C:/Users/ADMIN/Documents/MiniConda/DACNTT2', còn
# StandardScaler nằm trong 'C:/Users/ADMIN/Documents/MiniConda/DACNTT2/Scaler' thì ta sử dụng đường dẫn sau:
sc_X = joblib.load('./Scaler/sc_X.pkl')
sc_y = joblib.load('./Scaler/sc_y.pkl')
# Tương tự với khoảng giá min, max, phổ biến nhất:
kv_px_KhoangGiaDict = joblib.load('./Data/kv_px_KhoangGiaDict.pkl')
# Dự án/Số Hẻm Đường:
kv_px_shdDict = joblib.load('./Data/kv_px_shdDict.pkl')
# Và Encoder:
onehot_encoder = joblib.load('./Encoder/onehot_encoder.pkl')
label_encoder_px = joblib.load('./Encoder/label_encoder_px.pkl')
label_encoder_shd = joblib.load('./Encoder/label_encoder_shd.pkl')

# Tùy theo value về tên mô hình nhận được từ form của user submit mà gán model tương ứng:
def find_model_by_model_name(model_name):
    if (model_name == "model_Lasso"):
        model = joblib.load('./Model/model_Lasso.pkl')
        print("model_Lasso loaded---")
    elif (model_name == "model_RFR"):
        model = joblib.load('./Model/model_RFR.pkl')
        print("model_RFR loaded---")
    elif (model_name == "model_SVR"):
        model = joblib.load('./Model/model_SVR.pkl')
        print("model_SVR loaded---")
    elif (model_name == "model_XGB"):
        model = joblib.load('./Model/model_XGB.pkl')
        print("model_XGB loaded---")
    else:
        model = joblib.load('./Model/model_RidgeR.pkl') # Mặc định sẽ chọn mô hình Ridge.
        print("model_RidgeR loaded---")
    return model

def find_shd_by_random(Khu_vuc, phuong_xa):
    diaphuongName = Khu_vuc + "_" + phuong_xa.replace("Phường ", "")
    shd = random.choice(kv_px_shdDict[diaphuongName])
    print("dự án/số hẻm đường =", shd)
    return shd

def print_UserSubmitData(model_name, Khu_vuc, phuong_xa, Dien_tich, So_tang, So_phong_ngu, So_toilet, khoang_gia_pho_bien_min, gia_pho_bien_nhat, khoang_gia_pho_bien_max):
    print("Data type check:", type(model_name), "+", type(Khu_vuc), "+", type(phuong_xa), "+", type(Dien_tich), "+", type(So_tang), "+", type(So_phong_ngu), "+", type(So_toilet))
    print("Mô hình =", model_name)
    print("Diện tích =", Dien_tich, "; Số tầng =", So_tang, "; Số phòng ngủ =",  So_phong_ngu, "; Số toilet =", So_toilet)
    print("Khu vực =", Khu_vuc, "; Phường xã =", phuong_xa)
    print("Giá phổ biến Min =", khoang_gia_pho_bien_min, "; Giá phổ biến Nhất =", gia_pho_bien_nhat, "; Giá phổ biến Max =", khoang_gia_pho_bien_max)

# route() dùng để chỉ định cho Flask biết là URL nào sẽ kích hoạt hàm này.
# Trong trường hợp của hàm bên dưới, '/' nghĩa là khi mới mở web lên, hoặc thư mục root.
@app.route('/')
def home():
    return render_template('index.html')
    # Chạy ứng dụng web sẽ đưa chúng ta đến 'index.html'.
    # render_template nghĩa là tìm kiếm file 'index.html' có trong thư mục templates.

# Vận dụng đối số (argument) của route() để xử lý các phương thức HTTP khác nhau. Trong đó:
# GET: gửi một thông điệp GET và server sẽ trả về dữ liệu.
# POST: gửi một HTML form đến server. Ở đây, chúng ta sẽ dùng methods=['POST'] để cho phép user nộp một HTML form.
# Và sau đó chuyển hướng đến trang '/predict' có chứa kết quả dự đoán.
@app.route('/predict', methods=['POST'])
def predict():
    # Lấy dữ liệu từ form:
    model_name = str((request.form['model_name']).strip())
    model = find_model_by_model_name(model_name)
    Khu_vuc = str((request.form['Khu_vuc']).strip())
    phuong_xa = str((request.form['phuong_xa']).strip())
    Dien_tich = float(((request.form['Dien_tich']).strip()).replace(",", ".")) # Input mẫu: 6.1 hoặc 6,1.
    So_tang = int(round(float(((request.form['So_tang']).strip()).replace(",", ".")), 1)) # Input mẫu: 1 hoặc 2.
    So_phong_ngu = int(round(float(((request.form['So_phong_ngu']).strip()).replace(",", ".")), 1)) # Input mẫu: 2 hoặc 3.
    So_toilet = int(round(float(((request.form['So_toilet']).strip()).replace(",", ".")), 1)) # Input mẫu: 0 hoặc 1.
    
    # Lấy dữ liệu từ dictionary:
    keyname = Khu_vuc + "_" + phuong_xa.replace("Phường ", "")
    #print(Khu_vuc, "và", phuong_xa, "có khoảng giá:", kv_px_KhoangGiaDict[keyname])
    khoang_gia_pho_bien_min = min(kv_px_KhoangGiaDict[keyname])
    gia_pho_bien_nhat = statistics.mode(kv_px_KhoangGiaDict[keyname])
    khoang_gia_pho_bien_max = max(kv_px_KhoangGiaDict[keyname])
    print_UserSubmitData(model_name, Khu_vuc, phuong_xa, Dien_tich, So_tang, So_phong_ngu, So_toilet, khoang_gia_pho_bien_min, gia_pho_bien_nhat, khoang_gia_pho_bien_max)
    shd = find_shd_by_random(Khu_vuc, phuong_xa)
    
    # One-hot Encoding:
    col_names_toEncode = ['Khu vực']
    col_names_toRemove = ['phường/xã', 'dự án/số hẻm đường', 'Khoảng giá phổ biến Min', 'Giá phổ biến nhất', 'Khoảng giá phổ biến Max', 'Diện tích', 'Số tầng', 'Số phòng ngủ', 'Số toilet']
    data = [[Khu_vuc, phuong_xa.replace("Phường ", ""), shd, khoang_gia_pho_bien_min, gia_pho_bien_nhat, khoang_gia_pho_bien_max, Dien_tich, So_tang, So_phong_ngu, So_toilet]]
    df_base = pd.DataFrame(data, columns = (col_names_toEncode + col_names_toRemove)) # Learning data.
    #print(type(df_base))
    #print(df_base)
    df_encoding = (df_base.copy()).drop(columns=col_names_toRemove) # Get only data for encoding.
    df_encoding = pd.DataFrame(data=onehot_encoder.transform(df_encoding).toarray(), columns=onehot_encoder.get_feature_names_out(col_names_toEncode), dtype=bool)
    df_encoding = df_encoding * 1 # Convert "True" hoặc "False" thành giá trị 1 hoặc 0 tương ứng.
    data_final = pd.concat((df_base[col_names_toRemove], df_encoding), axis=1) # Nối 2 dataframe lại với nhau.
    # Label Encoding:
    data_final['phường/xã'] = label_encoder_px.transform(data_final['phường/xã'])
    data_final['dự án/số hẻm đường'] = label_encoder_shd.transform(data_final['dự án/số hẻm đường'])
    
    X_scaled = sc_X.transform(data_final) # Feature Scaling.
    prediction = model.predict(X_scaled.reshape(1, -1)) # X_scaled hoặc features phải có dạng: [[a, b]].
    prediction_unscaled = int(sc_y.inverse_transform(prediction.reshape(1, -1))[0]) # Convert về dạng số, thay vì sử dụng tiếp Numpy Array.
    #print("prediction_unscaled có dạng:", type(prediction_unscaled))
    output = float(abs(round(prediction_unscaled, 2))) # Lấy số dương và làm tròn mức giá bán nhà. Ví dụ: 61400.591732 -> 61400
    # Tên gọi "prediction_text" trong backend và trong html phải y hệt nhau:
    # return render_template('index.html', prediction_text='Giá dự đoán: {} tỷ'.format(round(output/1000, 1))) # Convert triệu sang tỷ, ví dụ: 4200 (triệu) -> 4.2 (tỷ).
    return render_template("cost-prediction.html", prediction_text=round(output/1000, 1))


if __name__ == "__main__":
    app.run(debug=True)
    # Hàm Main, nếu bất kỳ lỗi nào xảy ra, thì chúng sẽ hiện lên terminal.