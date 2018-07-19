import pickle

import numpy as np

from data_utils import constants, get_chunks, transform_data
import tensorflow as tf


def get_model_api():
    with open('server/saved_model/word_tokenizer.pkl', 'rb') as file:
        word_tokenizer = pickle.load(file)
    with open('server/saved_model/char_tokenizer.pkl', 'rb') as file:
        char_tokenizer = pickle.load(file)

    sess = tf.Session()
    meta_graph_def = tf.saved_model.loader.load(
        sess,
        [tf.saved_model.tag_constants.SERVING],
        'server/saved_model'
    )
    signature = meta_graph_def.signature_def
    word_ids = sess.graph.get_tensor_by_name(
        signature['sequence_tags'].inputs['word_ids'].name)
    char_ids = sess.graph.get_tensor_by_name(
        signature['sequence_tags'].inputs['char_ids'].name)
    sequence_length = sess.graph.get_tensor_by_name(
        signature['sequence_tags'].inputs['sequence_length'].name)
    word_length = sess.graph.get_tensor_by_name(
        signature['sequence_tags'].inputs['word_length'].name)
    decode_tags = sess.graph.get_tensor_by_name(
        signature['sequence_tags'].outputs['decode_tags'].name)
    best_scores = sess.graph.get_tensor_by_name(
        signature['sequence_tags'].outputs['best_scores'].name)

    def predict(texts):
        transformed = [
            transform_data.transform_data(text, word_tokenizer, char_tokenizer) for text in texts
        ]
        seq_len = [x[1] for x in transformed]
        words = [x[0] for x in transformed]
        chars = [x[2] for x in transformed]
        word_lengths = transform_data.pad_sequences(
            [x[3] for x in transformed], max(seq_len))
        max_char_len = np.max(word_lengths)
        padded_chars = np.zeros([len(texts), max(seq_len), max_char_len])
        for p1, c1 in zip(padded_chars, chars):
            for i, c2 in enumerate(c1):
                p1[i][:len(c2)] = c2
        feed_dict = {
            word_ids: transform_data.pad_sequences(words, max(seq_len)),
            sequence_length: seq_len,
            char_ids: padded_chars,
            word_length: word_lengths
        }
        predicted = sess.run([decode_tags, best_scores], feed_dict=feed_dict)
        origin_words = (x[4] for x in transformed)
        return [
            {
                "tags": [
                    {
                        "content": " ".join(x[0][s:e]),
                        "type":constants.REVERSE_TAGS[t]
                    } for t, s, e in get_chunks.get_chunks(x[1], constants.CLASSES)
                ],
                "score": float(x[2])
            }
            for x in zip(origin_words, predicted[0], predicted[1])
        ]
    return predict


if __name__ == "__main__":
    texts = [
        """Bán nhà 88 / 18 . . . Nguyễn Văn Qùy , phường Phú thuận , Q 7 . Diện tích đất 5 x 9 m . Diện tích sử dụng 130 m 2 ; xây dựng 1 trệt , 1 lửng , 1 lầu với 3 phòng , 2 WC , phòng khách và bếp . Khu dân cư an ninh , gần chợ , trường đại học , bệnh viện , khu trung tâm , cách Phú Mỹ Hưng 1 km , gần Cầu Phú Mỹ , cách siêu thị Điên Máy Xanh 300 m . 
Nhà mới , thiết kế hiện đại , nội thất cao cấp . Nhà hướng Đông Nam đón nắng sớm và gió mát . Các phòng thông tầng nhận ánh sáng tự nhiên . 
Giá 2 , 6 tỷ . LH : C . PHƯƠNG ( 0938 58 1238 )""",
        """Bán đất tiện xây phòng trọ cho thuê, gần nhà máy sữa Vinamilk, kumho, colgate với hơn 35000 công nhân đang làm việc ở đây, sát trường đại học quốc tế miền đông

Bán đất xây nhà trọ Bình Dương vị trí rất đẹp, đường xá rộng lớn xe hơi đổ cửa, xung quanh dân cư sinh sống rất đông, buôn bán tấp nập, rất thích hợp kinh doanh buôn bán, xây kiot, quán ăn,...... 
DT: 24mx30m= 720m2, sổ đỏ riêng đã tách 4 sổ riêng.


Giá: 450 triệu/sổ.

Đất sổ đỏ - thổ cư 100%, đường đã trải nhựa
Vui lòng liên hệ chính chủ: 0903 995 824 - 0902 969 278""",
        """Bán gấp trong năm nhà 2 MT đường Đoàn Thị Điểm, P1, Phú Nhuận 
Vị trí: Cách MT Phan Đăng Lưu chỉ 40m, cách ngã tư Phú Nhuận 100m. Nằm khu vực trung tâm, cách các tiện ích cần thiết chỉ vài phút đi bộ. 
Kết cấu: Nhà 1T, 1L cũ nhưng nội thất đẹp, góc 2 MT dễ kinh doanh buôn bán 
DT: 4.25x13m, đất vuông vức, không lộ giới 
Pháp lý: Sổ hồng chính chủ, đầy đủ pháp lý, sổ mới năm 2017 
Giá bán: 12 tỷ, thương lượng chút xíu lấy lộc. 
LH xem nhà chính chủ 0967463475 (Mr. Hóa)""",
        """Vị trí: Cách chợ Bình Chánh 3km
Tọa lạc tại mặt tiền đường liên khu KCN Cầu Tràm và đường Đinh Đức Thiện nối dài(DT 836)
Đối diện KCN Cầu Tràm không khói với gần 30.000 cán bộ,chuyên gia, công nhân viên Cầu Tràm đang sinh sống và làm việc
Địa Thế Tuyệt Vời – Dễ Dàng Kết Nối 
2 phút đến với KCN Cầu Tràm quy mô 80ha, chợ Cầu Tràm, các khu ẩm thực, nhà hàng, trung tâm vui chơi giải trí, trạm xăng, xe bus
5 phút đến với trường học các cấp, bệnh viện, UBND, trung tâm y tế, ngân hàng. cao tốc Bến Lức – Long Thành. 
+ 15 phút để kết nối trực tiếp với các tuyến giao thông huyết mạch như Quốc Lộ 1, Đại lộ Nguyễn Văn Linh, Đại lộ Võ Văn Kiệt, đến với siêu thị, chợ Bình Chánh, bến xe Miền Tây mới, Bệnh viện Nhi đồng 3. 
Tiếp giáp với nhiều tuyến giao thông huyết mạch về Tiền Giang, ra Quốc lộ 50, đến trung tâm TP HCM chỉ từ 15 – 20 phút 
Pháp lý: đảm bảo sổ đỏ thổ cư riêng từng nền 100%. 
thuận tiện mua ở - kinh doanh nhà trọ - đầu tư sinh lời
    """,
        """Cần bán gấp trước tết nhà Hẻm xe hơi.
- Diện tích:4*12 nở hậu 7.5m nhà còn mới cực đẹp dọn vào ở ngay
- Vị Trí: cực đẹp,hẻm to,an ninh,khu dân trí cao,..
- Xung quanh đầy đủ các tiện ích:chợ,trường học,trung tâm giải trí sầm uất nhất quận,...
- Nhà có sân trước rộng,đỗ xe hơi thoải mái.
LH:Hoàng Vũ (24/24h) 
Đ.C: Đường Nguyễn Văn Trỗi.""",
        """
– Diện tích: 300m2 = 10 x 30, giá bán 315 triệu/nền ( có 2 nền )
– Diện tích: 300m2 = 10 x 30, giá bán 435 triệu/nền ( có 2 nền )
- Đất gần chợ, trường học, nhà trẻ, công viên và KCN Nhật – Sing. 
Dân cư đông đúc, đất tiện kinh doanh, xây nhà trọ cho thuê ngay.

Tất cả đất giáp Tp Hồ Chí Minh đều có:
- Sổ hồng riêng, thổ cư 100% ==> Giao sổ hồng và đất ngay.
- Mua bán tại phòng công chứng nhà nước
- Bao sang tên và các thủ tục giấy tờ.

Liên hệ ngay: 0903 995 824 - 0979 137 937""",
        """Giá chỉ từ 2tr/m2 đến 4tr/m2 sở hữu ngay đất nền mặt tiền đường 16m-62m, vị trí đắc địa, dân cư đông đúc, xung quanh có đầy đủ các tiện ích, dịch vụ. Cơ sở hạ tầng hoàn thiện, sử dụng được ngay. 
Thông Tin Một Số Lô Đất 
+ DT: 5×30m = 150m2 – Giá 300tr, đối diện KCN đang hoạt động. 
+ DT: 5×30m = 150m2 – Giá 450tr, ngay TTHC mới quận. 
+ DT: 6×25m = 150m2 – Giá 520tr, gần chợ, khu dân cư đông đúc. 
+ DT: 9x30m = 270m2 – Giá 680tr, MT đường 16m. Tiện ở, xây trọ. 
+ DT: 10x30m = 300m2 – Giá 780tr, gần khu thể thao rộng lớn, bệnh viện. 
+ DT: 10x30m = 300m2 – Giá 850tr, kế góc TTTM, rất tiềm năng. 
+ DT: 20x30m = 600m2 – Giá 1 tỷ 300tr, trên khu đồi biệt thự, thích hợp nghỉ dưỡng. 
+ DT: 20x30m = 600m2 – Giá 1 tỷ 500tr, vị trí đẹp khu dân cư đông, gần chợ, trường học.""",
        "Mua nhà mặt tiền đường Võ Văn Tần tiện kinh doanh.",
        "Mình có nhu cầu mua nhà mặt tiền đường Võ Văn Tần tiện kinh doanh.",
        "Mình cần thuê nhà 1 trệt 1 lầu có phòng ngủ và PK đường Nguyễn Đình Chiểu"
    ]
    for v in get_model_api()(texts):
        print(v)
