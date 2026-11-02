from pathlib import Path

# Đường dẫn
DATA_DIR = Path("data/raw-img")   # thư mục sau khi giải nén Animals-10
OUT_DIR = Path("outputs")         # checkpoint, biểu đồ, báo cáo
OUT_DIR.mkdir(exist_ok=True)

# Chung
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 64
NUM_WORKERS = 2

# Tên thư mục trong Animals-10 là tiếng Ý -> đổi sang tiếng Anh để hiển thị
TRANSLATE = {
    "cane": "dog",
    "cavallo": "horse",
    "elefante": "elephant",
    "farfalla": "butterfly",
    "gallina": "chicken",
    "gatto": "cat",
    "mucca": "cow",
    "pecora": "sheep",
    "ragno": "spider",
    "scoiattolo": "squirrel",
}
