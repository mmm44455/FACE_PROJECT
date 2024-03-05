Cách làm để nhận diện khuôn mặt theo phương pháp CNN

Đầu tiên ta phải cài đặt một số thư viện : 
    matplotlib
    tensorflow
    keras
    os
    opencv-python
    haarcascade_frontalface_default.xml
    facenet_pytorch,MTCNN

    
B1:
Thu thập dữ liệu khuôn mặt để huấn luyện vì vậy để có dữ liệu thì tôi chụp ảnh khuôn mặt bằng file chup_anh.py 

    Với file này nó sẽ chụp chỉ riêng khuôn mặt bạn thôi bằng vào nó bạn có thể thu thập nhieeuf dữ liệu với nhiều góc chụp và nhiều khuôn mặt hơn 
    
    Gồm 2 file : 1 file ảnh và 1 file để test lại sau quá trình train 

    
![img](https://github.com/mmm44455/FACE_PROJECT/assets/132626865/a6c40366-7368-4b9d-a6dd-957583316d75)

B2:
Sau khi thu thập đầy đủ ta tiến đên bước tạo model cho chương trình bằng model_tao.py 

 Với chương trình này tôi có thể tạo model và lưu nó lại sau đó show hình ảnh biểu đồ độ chính xác và sai số trong quá trình train
B3:
Sau khi tạo model xong ta đến bước test thực nghiệm qua chương trình test.py dùng bằng cam

Với haarcascade_frontalface_default.xml dùng để phát hiện khuôn mặt còn model để xem khuôn mặt này là của ai 
