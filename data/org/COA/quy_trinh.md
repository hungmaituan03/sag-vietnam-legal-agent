# Quy trình tín dụng bán lẻ (stub — fake fixture)

**Không phải quy chế thật. Chỉ dùng cho test / demo.**

## 1. Tiếp nhận hồ sơ
- Khách hàng đăng ký qua ứng dụng hoặc điểm bán đối tác.
- Thu thập thông tin định danh (KYC) và đồng ý tra cứu CIC (mô tả stub).

## 2. Thẩm định tự động
- Hệ thống chấm điểm tín dụng nội bộ (auto-score).
- Nếu đạt và số tiền ≤ 500 triệu VND → đề xuất phê duyệt trong hạn mức.
- Nếu điểm cận biên hoặc số tiền > 500 triệu VND → chuyển Hội đồng tín dụng.

## 3. Hội đồng tín dụng
- Thành phần (fixture): 3 thành viên do TGĐ chỉ định.
- Họp / lấy ý kiến bằng văn bản; chỉ giải ngân khi có đủ phiếu đồng ý theo quy chế nội bộ (stub).

## 4. Giải ngân
- Chỉ giải ngân sau khi hợp đồng điện tử được ký kết.
- Giải ngân về tài khoản khách hàng hoặc đối tác bán hàng theo sản phẩm.

## 5. Thu hồi nợ (tóm tắt)
- Ngày 1–30 quá hạn: nhắc nhẹ (tin nhắn / gọi tự động).
- Từ ngày 31: bàn giao bộ phận thu hồi nợ.

## 6. Rào chắn rủi ro (fake rules)
- Tỷ lệ nợ trên thu nhập khai báo (DTI) tối đa 40%.
- Không cấp khoản mới nếu khách có từ 2 lần chậm trả trở lên trong 6 tháng gần nhất.
- Cán bộ / người lao động Alpha Finance vay vốn: cần chữ ký TGĐ và Phó TGĐ rủi ro.

Ghi chú: file stub bổ sung cho merge evidence / Q&A entity-grounded. Không dùng cho tư vấn thật.
