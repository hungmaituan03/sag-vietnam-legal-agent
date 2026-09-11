# What SAG adds over the Voyage shortlist

Generated 2026-09-11 03:44 UTC by `scripts/run_sag_evidence.py` against the approved
finance corpus (4098 chunks, 19359 semantic edges).

Settings: hybrid-k=12, voyage-k=5, sag-extra=30, min-sim=0.7, max-sim=0.99.

`Orphans` counts reranked seeds that are fragments whose parent text was
absent from the shortlist. `Repaired` counts how many SAG recovered.
`New laws` are statutes no seed came from, reached by a semantic edge.

| Query | Seeds | With SAG | Orphans | Repaired | New laws |
| --- | --- | --- | --- | --- | --- |
| kỳ kế toán năm | 5 | 31 | 4 | 4 | law-2020-luat-doanh-nghiep |
| điều kiện cấp giấy phép thành lập tổ chức tín dụng | 5 | 35 | 5 | 5 | — |
| kiểm kê tài sản | 5 | 33 | 1 | 1 | law-2020-luat-doanh-nghiep, luat-ngan-hang-nha-nuoc |
| báo cáo tài chính hằng năm | 5 | 35 | 5 | 5 | — |
| nhiệm vụ và quyền hạn của Ngân hàng Nhà nước | 5 | 35 | 2 | 2 | luat-cac-to-chuc-tin-dung |
| vốn điều lệ của tổ chức tín dụng | 5 | 35 | 5 | 5 | — |
