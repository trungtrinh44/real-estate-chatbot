import argparse
import json
import os
import pickle
import shutil

from data_utils import (constants, get_chunks, process_train_data, read_data,
                        read_word_vec, transform_data)
from model import configs, ner_model


def predict(text):
    words, seq_len, chars, char_lens, origin_words = transform_data.transform_data(
        text, word_tokenizer, char_tokenizer)
    predicted = model.predict_batch(
        sentences=[words],
        sentence_lengths=[seq_len],
        words=[chars],
        word_lengths=[char_lens],
    )
    # _words = [[idx2word[x] for x in seq] for seq in words]
    # _tags = [[idx2tag[x] for x in seq] for seq in predicted[0]]
    return [(' '.join(o[s:e]), t) for o, p in zip([origin_words], predicted[0]) for t, s, e in get_chunks.get_chunks(p, constants.CLASSES)], predicted[1][0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='./output/test')
    parser.add_argument('--model', type=str)
    parser.add_argument('--version', type=str)
    args = parser.parse_args()
    model = ner_model.load_model(os.path.join(
        args.model, 'model-{}'.format(args.version)))
    model.configs.vocab_tags = constants.CLASSES
    with open(os.path.join(args.model, 'word_tokenizer.pkl'), 'rb') as file:
        word_tokenizer = pickle.load(file)
    with open(os.path.join(args.model, 'char_tokenizer.pkl'), 'rb') as file:
        char_tokenizer = pickle.load(file)
    filenames = []
    for root, _, files in os.walk(args.input):
        for file in files:
            if file.endswith('.txt'):
                filenames.append(os.path.join(root, file))
    test_iter = read_data._create_iterator(filenames, 64, 0)
    result = model.evaluate_step(test_iter)
    # shutil.rmtree('temp', ignore_errors=True)
    import pandas as pd
    data = pd.DataFrame.from_dict(result, 'index')
    print(data)
    data.to_csv(os.path.join(args.model, 'eval-{}.csv'.format(args.version)))
    text="Cần bán nhà phố 2 lầu, ST hẻm 33, Đường số 1, Lý Phục Man, phường BT\nCần bán nhà phố 2 lầu, ST hẻm 33, Đường số 1, Lý Phục Man, phường Bình Thuận, Quận 7.\n- DT : 4x12, hướng Đông\n- Kết cấu: Trệt, 2 lầu, ST, 4 PN, WC riêng.\n- Vị trí đẹp, hẻm xe hơi 6m, khu dân cư đông đúc, sầm uất, thích hợp KD mở VPCTY.\n- Ngay TT quận 7, giao thông thuận tiện, cách các quận TT 2-4km.\n- Pháp lý : SH hoàn công"
    print(predict(text))
    text="Cần mua nhà có 6 phòng ngủ diện tích từ 10 m 2 - 200 m 2"
    print(predict(text))
    text="Cần mua nhà có 6 phòng ngủ diện tích từ 10 m 2 đến 200 m 2"
    print(predict(text))
    text="Cần mua nhà có 6 phòng ngủ diện tích trên 10 m 2"
    print(predict(text))
    text = """Do mình chuẩn bị chuyển vào ktx nên phòng mình đang ở còn trống 1 chổ.
Tiền phòng: 1tr850/tháng, ở được 3 người (hiện tại có 2 người, đã ra trường đi làm), có tủ lạnh.
Địa chỉ: 58/7A1. đường Đồng Nai, CX bắc hải. cách cổng 3 THT tầm 300m.
Giờ giấc: 6h00-22h30, về trễ nhớ báo trước.
Điện nước theo giá nhà nước, Wifi (mới lắp mới, ping 8ms) 30k/tháng.
Bạn nào có nhu cầu thì inbox mình (0932069143)"""
    print(predict(text))
    text = """Cơ sở hạ tầng đầy đủ. Xung quanh nhiều tiện ích."""
    print(predict(text))
    text = """Bán đất tiện xây phòng trọ cho thuê, gần nhà máy sữa Vinamilk, kumho, colgate với hơn 35000 công nhân đang làm việc ở đây, sát trường đại học quốc tế miền đông

Bán đất xây nhà trọ Bình Dương vị trí rất đẹp, đường xá rộng lớn xe hơi đổ cửa, xung quanh dân cư sinh sống rất đông, buôn bán tấp nập, rất thích hợp kinh doanh buôn bán, xây kiot, quán ăn,......
DT: 24mx30m= 720m2, sổ đỏ riêng đã tách 4 sổ riêng.


Giá: 450 triệu/sổ.

Đất sổ đỏ - thổ cư 100%, đường đã trải nhựa
Vui lòng liên hệ chính chủ: 0903 995 824 - 0902 969 278"""
    print(predict(text))
    text = """Cần thuê nhà để kinh doanh bún phở. Mặt đường hoặc ngõ to, có chỗ để xe tại quận Hai Bà Trưng, Thanh Xuân, Hoàng Mai hoặc Đống Đa. Diện tích >=25m2, giá thuê 8 - 12 triệu, thanh toán 3 tháng /lần."""
    print(predict(text))
    text = """Bán gấp trong năm nhà 2 MT đường Đoàn Thị Điểm, P1, Phú Nhuận
Vị trí: Cách MT Phan Đăng Lưu chỉ 40m, cách ngã tư Phú Nhuận 100m. Nằm khu vực trung tâm, xung quanh đầy đủ các tiện ích. Cơ sở hạ tầng đầy đủ.
Kết cấu: Nhà 1T, 1L cũ nhưng nội thất đẹp, góc 2 MT dễ kinh doanh buôn bán
DT: 4.25x13m, đất vuông vức, không lộ giới
Pháp lý: Sổ hồng chính chủ, đầy đủ pháp lý, sổ mới năm 2017
Giá bán: 12 tỷ, thương lượng chút xíu lấy lộc.
LH xem nhà chính chủ 0967463475 (Mr. Hóa)"""
    print(predict(text))
    text = """Vị trí: Cách chợ Bình Chánh 3km
Tọa lạc tại mặt tiền đường liên khu KCN Cầu Tràm và đường Đinh Đức Thiện nối dài(DT 836)
Đối diện KCN Cầu Tràm không khói với gần 30.000 cán bộ,chuyên gia, công nhân viên Cầu Tràm đang sinh sống và làm việc
Địa Thế Tuyệt Vời – Dễ Dàng Kết Nối
2 phút đến với KCN Cầu Tràm quy mô 80ha, chợ Cầu Tràm, các khu ẩm thực, nhà hàng, trung tâm vui chơi giải trí, trạm xăng, xe bus
5 phút đến với trường học các cấp, bệnh viện, UBND, trung tâm y tế, ngân hàng. cao tốc Bến Lức – Long Thành.
+ 15 phút để kết nối trực tiếp với các tuyến giao thông huyết mạch như Quốc Lộ 1, Đại lộ Nguyễn Văn Linh, Đại lộ Võ Văn Kiệt, đến với siêu thị, chợ Bình Chánh, bến xe Miền Tây mới, Bệnh viện Nhi đồng 3.
Tiếp giáp với nhiều tuyến giao thông huyết mạch về Tiền Giang, ra Quốc lộ 50, đến trung tâm TP HCM chỉ từ 15 – 20 phút
Pháp lý: đảm bảo sổ đỏ thổ cư riêng từng nền 100%.
thuận tiện mua ở - kinh doanh nhà trọ - đầu tư sinh lời
    """
    print(predict(text))
    print(predict("""Cần bán gấp trước tết nhà Hẻm xe hơi.
- Diện tích:4*12 nở hậu 7.5m nhà còn mới cực đẹp dọn vào ở ngay
- Vị Trí: cực đẹp,hẻm to,an ninh,khu dân trí cao,..
- Xung quanh đầy đủ các tiện ích:chợ,trường học,trung tâm giải trí sầm uất nhất quận,...
- Nhà có sân trước rộng,đỗ xe hơi thoải mái.
LH:Hoàng Vũ (24/24h)
Đ.C: Đường Nguyễn Văn Trỗi."""))
    print(predict("""
– Diện tích: 300m2 = 10 x 30, giá bán 315 triệu/nền ( có 2 nền )
– Diện tích: 300m2 = 10 x 30, giá bán 435 triệu/nền ( có 2 nền )
- Đất gần chợ, trường học, nhà trẻ, công viên và KCN Nhật – Sing.
Dân cư đông đúc, đất tiện kinh doanh, xây nhà trọ cho thuê ngay.

Tất cả đất giáp Tp Hồ Chí Minh đều có:
- Sổ hồng riêng, thổ cư 100% ==> Giao sổ hồng và đất ngay.
- Mua bán tại phòng công chứng nhà nước
- Bao sang tên và các thủ tục giấy tờ.

Liên hệ ngay: 0903 995 824 - 0979 137 937"""))
    print(predict("Mua nhà mặt tiền đường Võ Văn Tần tiện kinh doanh."))
    print(predict("Mình có nhu cầu mua nhà mặt tiền đường Võ Văn Tần tiện kinh doanh."))
    print(predict(
        "Mình cần thuê nhà 1 trệt 1 lầu có phòng ngủ và PK đường Nguyễn Đình Chiểu"))
    print(predict(
        "Cần mua đất ở Đà Nẵng gần các kcn để tiện xây phòng trọ."
    ))
    print(predict(
        """Người bạn Tôi có căn nhà mới xây đã hoàn thành và đang cập nhật hoàn công.
Vị trí: Ngay đường Bưng Ông Thoàn cách Aeon Mall khoảng 100 mét.
✔️ N 6,5 x D 17,07
✔️1 trệt 2 lầu
✔️6 phòng
✔️nhà làm cửa gỗ gõ đỏ
✔️ Giá: 5,150 tỷ. Giá áp dụng cho người thiện chí mua trong tuần này và tặng bộ bàn ghế như hình nhé khách yêu.
📱0933 146 038"""
    ))
    print(predict("""[NHÀ TRỌ]
Xin phép admin,
Hi all,
[1] Cuối tháng này (tầm 27 28 tháng 3), chổ mình có dư 2 phòng, tầm 17m2 ở được 2 người (nam nữ có thể ở chung), nấu ăn thoải mái, điện nước theo giá nhà nước, wifi 30k/tháng (mới thay mạng đầu năm, ping 8ms nếu cắm cáp)
Giá phòng: 1tr7
[2] 1 Slot ở ghép NAM với 2 người đi làm cả ngày tối mới về (1 bác sĩ, 1 IT), 2tr/tháng/3người
Điện, nước, wifi như trên, phòng này có tủ lạnh.
Có thể dọn vô ở liền.
--
Địa chỉ: Hẻm 58/7A1, đường Đồng Nai, Q.10 (cách cổng 3 THT tầm 500m).
Để biết thêm chi tiết vui lòng liên hệ or inbox: 0932 069 143
Số chủ nhà: 0909 088 688 (cụ An)
Thanks."""))
    print(predict("""Hi mọi người, em đang cần thuê một cái xưởng tầm 1000-1500m2. Khu vực lân cận xã Tân Xuân, hóc môn. Ai biết giới thiệu em với nhé. Rất cám ơn mọi người. Chúc mọi người một ngày tốt lành! 0906786406 Mr. Tâm"""))
    print(
        predict(
            """Tòa nhà 4 tầng cho thuê văn phòng ảo ,văn phòng chia sẻ 
Tòa nhà mặt tiền đường Võ Văn Kiệt - Lý Đạo Thành , giao thông thuận lợi giữa các Quận 

Gói 1 - 300.000 đ/ tháng ( đặt bảng hiệu, chỗ ngồi linh hoạt 2 ngày /tuần , hỗ trợ dịch vụ đăng ký giấy phép ) . Thanh toán 12 tháng tặng 2 tháng 
Gói 2 - 600.000đ/ tháng ( đặt bảng hiệu, chỗ ngồi linh hoạt 3 ngày /tuần , hỗ trợ dịch vụ đăng ký giấy phép. Miễn phí hosting ,website ) . Thanh toán 12 tháng tặng 2 tháng 
Gói 3 - 1.000.000đ/ tháng ( đặt bảng hiệu, chỗ ngồi cố định 5 ngày /tuần , hỗ trợ dịch vụ đăng ký giấy phép. Miễn phí hosting ,website, tổng đài ) . Thanh toán 12 tháng tặng 2 tháng
Gói 4 - 1.500.000đ/ tháng ( đặt bảng hiệu, chỗ ngồi cố định 5 ngày /tuần , hỗ trợ dịch vụ đăng ký giấy phép. Miễn phí hosting ,website, tổng đài, kế toán báo cáo thuế hàng toán ) . Thanh toán 12 tháng tặng 2 tháng
Liên hệ 0909237107"""
        )
    )
    print(
        predict(
            """Cho thuê nhà nguyên căn mặt tiền đường Nguyễn Trọng Tuyển, quận Tân Bình.
+ Nhà rộng 4,5 m dài 24m
+ Nhà có 1 trệt, 1 lửng, 3 lầu. có 6 phòng, tolet riêng mỗi phòng 
+ Trệt thiết kế trống suốt, cầu thang cuối nhà.
+ Nhà có hầm để được 30 chiếc xe máy và khoảng sân trước rộng, có thang máy.
+ Thích hợp kinh doanh đa ngành nghề, quán cà phê, văn phòng đại diện, kinh doanh thời trang, phòng khám,....
+ Giá chỉ 55 triệu/tháng"""
        )
    )
    print(
        predict(
            """Diện tích 44m2
Vị trí: Đường Đỗ Pháp Thuật,An Phú,Quận 2
Thiết kế: MB 2 mặt tiền đường 28 và ĐPT 
thuận tiện làm văn phòng, showroom,
Khuôn viên rất rộng và thoáng .
Khu an ninh cao , có nhiều văn phòng , 
Căn nhà phù hợp để mở văn phòng , kinh doanh, 
Chính Chủ ( miễn trung gian ) 
Vui lòng liên hệ để biết thêm thông tin:
Nguyễn Thủy: 090-139-6167"""
        )
    )
    print(
        predict(
            """Vì cần chuyển văn phòng, chúng tôi không cho thuê lại văn phòng rộng khoảng 40m2, văn phòng tại lầu 4, có thang máy, có 2 toa let riêng biệt, 1 phòng giám đốc rộng khoảng 10m2 có điều hòa đầy đủ, phòng còn lại rộng khoảng 30m2 có 2 điều hòa,phòng mới, sạch sẽ, view thoáng, có ánh sáng mặt trời. điện tính theo đồng hồ riêng, nước được miễn phí.
địa chi: lầu 4, 20/1 Nguyễn Trường Tộ, Phường 12, Quận 4, TP HCM
liên hệ Ms Vuong - 0907272867"""
        )
    )
    print(
        predict("""Tôi muốn sang nhượng mặt bằng như sau.
- 4 x 18 m giá 1 tỷ 4, SH
-  Gia Canh đi vào 10m.
- cạnh trường học
- Về thành phố có vài phút.
Do gia đình tôi phai chuyển ra bắc nên bán nhanh.
Liên hệ tôi 0901544773.
** Ngoài ra nhà tôi còn 1 miếng ở Quốc lộ 50 mặt tiền, vị trí thuận lợi, bán rẻ luôn 7 tỷ""")
    )