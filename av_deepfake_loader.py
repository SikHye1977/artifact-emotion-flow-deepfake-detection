import os
import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

class AVDeepfakeDataset(Dataset):
    def __init__(self, root_dir, split="val"):
        """
        AV-Deepfake1M 데이터셋 로더
        :param root_dir: AV-Deepfake1M_RootFiles 폴더의 경로
        :param split: 'train', 'val', 'test' 중 하나
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.extracted_dir = self.root_dir / f"extracted_{split}"
        
        # 메타데이터 JSON 로드
        metadata_path = self.root_dir / f"{split}_metadata.json"
        
        # test 폴더는 보통 json 메타데이터 대신 test_files.txt를 사용할 수도 있습니다.
        # 여기서는 기본적으로 json이 있다고 가정하고 작성합니다.
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            print(f"[{split.upper()}] 메타데이터 로드 완료: 총 {len(self.metadata)}개 데이터")
        else:
            print(f"경고: {metadata_path} 파일을 찾을 수 없습니다.")
            self.metadata = []

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # 1. 메타데이터에서 1개 샘플 정보 가져오기
        sample_info = self.metadata[idx]
        
        # ⚠️ 중요: JSON 구조에 따라 아래 키('video_path', 'label' 등)는 실제 데이터에 맞게 수정해야 합니다.
        # 데이터셋마다 키 이름이 다를 수 있습니다 (예: 'file_name', 'target', 'manipulation_type' 등)
        
        # 임의의 예상 키값으로 작성된 예시입니다.
        relative_video_path = sample_info.get("video_path", "") 
        label = sample_info.get("label", 0) # 0: Real, 1: Fake
        
        # 2. 실제 영상의 절대 경로 완성하기
        video_full_path = self.extracted_dir / relative_video_path
        
        # 3. (선택) 여기서 cv2나 torchaudio를 사용해 영상/음성 텐서를 추출할 수 있습니다.
        # 현재는 경로와 라벨, 메타데이터 원본만 반환합니다.
        
        return {
            "video_path": str(video_full_path),
            "label": label,
            "metadata": sample_info
        }

# ==========================================
# 🧪 실행 테스트 코드
# ==========================================
if __name__ == "__main__":
    # 실행 위치(~/hsh/AIApplication)를 기준으로 한 데이터셋 최상위 폴더 경로
    DATASET_ROOT = "./AV-Deepfake1M_RootFiles"
    
    # 1. Validation 데이터셋 로드 테스트
    # (부분 다운로드 하셨더라도, json 파일 내역 중 실제로 다운받아진 파일만 필터링해서 쓰시면 됩니다.)
    val_dataset = AVDeepfakeDataset(root_dir=DATASET_ROOT, split="val")
    
    # 2. 데이터가 잘 로드되었는지 첫 번째 샘플 확인
    if len(val_dataset) > 0:
        first_sample = val_dataset[0]
        print("\n--- 첫 번째 샘플 데이터 구조 확인 ---")
        print(f"비디오 실제 경로: {first_sample['video_path']}")
        print(f"존재 여부: {os.path.exists(first_sample['video_path'])}")
        print(f"메타데이터 내용: {json.dumps(first_sample['metadata'], indent=2)}")
        print("-----------------------------------")