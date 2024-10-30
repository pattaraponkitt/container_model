import os

def print_directory_structure(path, prefix="", file_limit=5):
    """
    แสดงโครงสร้างของโฟลเดอร์และไฟล์ทั้งหมดใน path ที่กำหนด โดยจำกัดจำนวนไฟล์ที่แสดง
    
    Parameters:
    path (str): path ที่ต้องการดูโครงสร้าง
    prefix (str): prefix สำหรับการแสดงผลในรูปแบบ tree structure
    file_limit (int): จำนวนไฟล์สูงสุดที่จะแสดงในแต่ละโฟลเดอร์
    """
    # ตรวจสอบว่า path มีอยู่จริง
    if not os.path.exists(path):
        print(f"ไม่พบ path: {path}")
        return

    # แสดงชื่อโฟลเดอร์หรือไฟล์ปัจจุบัน
    print(prefix + os.path.basename(os.path.abspath(path)))
    
    # ถ้าเป็นโฟลเดอร์ ให้แสดงเนื้อหาด้านใน
    if os.path.isdir(path):
        # รับรายการไฟล์และโฟลเดอร์ทั้งหมด
        items = os.listdir(path)
        items.sort()  # เรียงลำดับตามตัวอักษร
        
        # แยกโฟลเดอร์และไฟล์
        folders = [item for item in items if os.path.isdir(os.path.join(path, item))]
        files = [item for item in items if os.path.isfile(os.path.join(path, item))]
        
        # จำกัดจำนวนไฟล์ที่จะแสดง
        if len(files) > file_limit:
            files = files[:file_limit]
            show_remaining = True
        else:
            show_remaining = False
            
        # รวมรายการที่จะแสดงทั้งหมด
        items_to_show = folders + files
        
        # วนลูปแสดงแต่ละรายการ
        for i, item in enumerate(items_to_show):
            item_path = os.path.join(path, item)
            # เป็นรายการสุดท้ายหรือไม่ (รวมถึงกรณีที่มีไฟล์เพิ่มเติมที่ไม่แสดง)
            is_last = (i == len(items_to_show) - 1) and not show_remaining
            
            if is_last:
                print_directory_structure(item_path, prefix + "└── ", file_limit)
            else:
                print_directory_structure(item_path, prefix + "├── ", file_limit)
        
        # แสดงจำนวนไฟล์ที่เหลือ (ถ้ามี)
        if show_remaining:
            remaining_count = len(files) - file_limit
            print(f"{prefix}└── ... และอีก {remaining_count} ไฟล์")

# ใช้งานฟังก์ชัน
target_path = r"H:\My Drive\btran\container_number"
print("โครงสร้างโฟลเดอร์:")
print_directory_structure(target_path, file_limit=5)
