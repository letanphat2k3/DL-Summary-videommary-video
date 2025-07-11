import h5py

# ✅ Đường dẫn đến file (chú ý: dùng raw string hoặc \\)
file_path = r'C:\WorkSpace\KhoaLuanTotNghiep\DSNet\datasets\eccv16_dataset_summe_google_pool5.h5'

with h5py.File(file_path, 'r') as f:
    print("📂 Các nhóm và dataset trong file:")
    
    def print_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"  📊 Dataset: {name}, shape: {obj.shape}, dtype: {obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"  📁 Group: {name}")
    
    f.visititems(print_structure)

    # ✅ Lấy danh sách tất cả key cấp 1
    all_keys = list(f.keys())
    print("\n📌 Danh sách key (video):")
    for i, key in enumerate(all_keys):
        print(f"  {i+1}. {key}")

    # ✅ Đọc dữ liệu từ key đầu tiên (ví dụ)
    example_key = all_keys[0]
    print(f"\n📥 Đọc dữ liệu từ key đầu tiên: {example_key}")
    data = f[example_key][:]

    print(f"📐 Kích thước dữ liệu: {data.shape}")
    print(f"🔍 Hiển thị 10 dòng đầu tiên:")
    for i in range(min(10, len(data))):
        print(f"{i+1}: {data[i]}")
