# pip install opencv-contrib-python
import cv2
import numpy as np
from enum import Enum, unique

# enum으로 디스크립터 정의
@unique
class Descriptor(Enum):
    SIFT = 0  # SIFT (Scale-Invariant Feature Transform) 디스크립터
    SURF = 1  # SURF (Speeded-Up Robust Features) 디스크립터
    ORB = 2   # ORB (Oriented FAST and Rotated BRIEF) 디스크립터
    BRIEF = 3 # BRIEF (Binary Robust Independent Elementary Features) 디스크립터
    FREAK = 4 # FREAK (Fast Retina Keypoint) 디스크립터
    LATCH = 5 # LATCH (Learned Arrangements of Three Patch Codes) 디스크립터
    DAISY = 6 # DAISY (Dense Adaptive Scale-Invariant Descriptor) 디스크립터
    HOG = 7   # HOG (Histogram of Oriented Gradients) 디스크립터
    AKAZE = 8 # AKAZE (Accelerated KAZE) 디스크립터

# 디스크립터를 활용한 매칭 함수
def matching(main, target, types):
    # 1. 특징점 추출
    match types:
        case Descriptor.SIFT.value:
            descriptor = cv2.SIFT_create()  # SIFT 디스크립터 생성
        case Descriptor.SURF.value:
            descriptor = cv2.xfeatures2d.SURF_create()  # SURF 디스크립터 생성
        case Descriptor.ORB.value:
            descriptor = cv2.ORB_create()  # ORB 디스크립터 생성
        case Descriptor.BRIEF.value:
            # BRIEF 디스크립터 생성
            star = cv2.xfeatures2d.StarDetector_create()  # 특징점 탐지
            brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()  # BRIEF 디스크립터
            keypoints_main = star.detect(main, None)
            keypoints_target = star.detect(target, None)
            keypoints_main, descriptor_main = brief.compute(main, keypoints_main)
            keypoints_target, descriptor_target = brief.compute(target, keypoints_target)
            return visualize_matching(main, target, keypoints_main, descriptor_main, keypoints_target, descriptor_target, types)
        case Descriptor.FREAK.value:
            # FREAK 디스크립터 생성
            star = cv2.xfeatures2d.StarDetector_create()  # 특징점 탐지
            freak = cv2.xfeatures2d.FREAK_create()  # FREAK 디스크립터
            keypoints_main = star.detect(main, None)
            keypoints_target = star.detect(target, None)
            keypoints_main, descriptor_main = freak.compute(main, keypoints_main)
            keypoints_target, descriptor_target = freak.compute(target, keypoints_target)
            return visualize_matching(main, target, keypoints_main, descriptor_main, keypoints_target, descriptor_target, types)
        case Descriptor.LATCH.value:
            # LATCH 디스크립터 생성
            latch = cv2.xfeatures2d.LATCH_create()
            keypoints_main, descriptor_main = latch.detectAndCompute(main, None)
            keypoints_target, descriptor_target = latch.detectAndCompute(target, None)
            return visualize_matching(main, target, keypoints_main, descriptor_main, keypoints_target, descriptor_target, types)
        case Descriptor.DAISY.value:
            # DAISY 디스크립터 생성
            daisy = cv2.xfeatures2d.DAISY_create()
            keypoints_main, descriptor_main = daisy.detectAndCompute(main, None)
            keypoints_target, descriptor_target = daisy.detectAndCompute(target, None)
            return visualize_matching(main, target, keypoints_main, descriptor_main, keypoints_target, descriptor_target, types)
        case Descriptor.HOG.value:
            # HOG 디스크립터 생성 및 계산
            hog = cv2.HOGDescriptor()
            descriptor_main = hog.compute(main)
            descriptor_target = hog.compute(target)
            return ''  # HOG는 직접 매칭이 아닌 히스토그램 기반 비교에 활용됨
        case Descriptor.AKAZE.value:
            descriptor = cv2.AKAZE_create()  # AKAZE 디스크립터 생성
        default:
            raise ValueError("Unknown descriptor type")

    # 디스크립터를 사용하여 특징점과 디스크립터 추출
    keypoints_main, descriptor_main = descriptor.detectAndCompute(main, None)
    keypoints_target, descriptor_target = descriptor.detectAndCompute(target, None)

    return visualize_matching(main, target, keypoints_main, descriptor_main, keypoints_target, descriptor_target, types)

# 매칭 결과 시각화 함수
def visualize_matching(main, target, keypoints_main, descriptor_main, keypoints_target, descriptor_target, types):
    # Brute-Force 매칭 알고리즘 생성
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptor_target, descriptor_main)  # 매칭 수행
    sorted_matches = sorted(matches, key=lambda x: x.distance)  # 거리 기준 정렬

    # 매칭 결과 시각화
    result = cv2.drawMatches(target, keypoints_target, main, keypoints_main, sorted_matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # 결과 이미지 출력
    cv2.imshow(str(types), result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return ''

if __name__ == '__main__':
    # 입력 이미지 경로 설정
    main_path = './data/main05.jpg'  # 메인 이미지 경로
    target_path = './data/target05.jpg'  # 타겟 이미지 경로

    # 이미지 읽기 (그레이스케일)
    main = cv2.imread(main_path, cv2.IMREAD_GRAYSCALE)
    target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)

    # 디스크립터 타입 설정 (SIFT 사용 예)
    types = Descriptor.SIFT.value  # 변경하여 테스트
    matching(main, target, types)
