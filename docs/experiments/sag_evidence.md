# What SAG adds over the Voyage shortlist

Generated 2026-09-11 10:52 UTC by `scripts/run_sag_evidence.py` against the approved
finance corpus (8644 chunks, 39997 semantic edges).

Settings: hybrid-k=20, voyage-k=5, sag-extra=30, min-sim=0.7, max-sim=0.99.

`Orphans` counts reranked seeds that are fragments whose parent text was
absent from the shortlist. `Repaired` counts how many SAG recovered.
`New laws` are statutes no seed came from, reached by a semantic edge.

| Query | Seeds | With SAG | Orphans | Repaired | New laws |
| --- | --- | --- | --- | --- | --- |
| kỳ kế toán năm | 5 | 35 | 4 | 4 | law-2020-luat-doanh-nghiep |
| điều kiện cấp giấy phép thành lập tổ chức tín dụng | 5 | 35 | 5 | 5 | — |
| kiểm kê tài sản | 5 | 33 | 1 | 1 | law-2005-luat-so-huu-tri-tue, law-2020-luat-doanh-nghiep, law-2022-luat-so-huu-tri-tue-sua-doi, luat-ngan-hang-nha-nuoc |
| báo cáo tài chính hằng năm | 5 | 35 | 5 | 5 | — |
| nhiệm vụ và quyền hạn của Ngân hàng Nhà nước | 5 | 35 | 3 | 3 | — |
| vốn điều lệ của tổ chức tín dụng | 5 | 35 | 5 | 5 | — |
