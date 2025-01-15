# pip install opencv-contrib-python
# 1. enum State(복잡한 조건을 하나의 '상태'로 만들어서 쉽게 분류)
import cv2
import numpy as np
from enum import Enum, unique

# 데코레이터 unique
# Descriptor라는 enum state안의 상태가 '중복되지 않도록' 하는 역할
@unique
class Descriptor(Enum):
    SIFT = 0
    HOG = 1
    SURF = 2
    ORB = 3
    
# 디스크립터를 활용한 패턴 비교
def matching(main, target, types):
    
    # 1. 특징점 추출
    # switch case 문으로 구현
    match types: 
        case 0 : 
            descriptor = cv2.SIFT_create()
        case 3 : 
            descriptor = cv2.ORB_create()
    
    
    # 해당 디스크립터로 -> 대상 이미지에서 특징점을 추출(detect -> compute)
    keypoints_main, descriptor_main = descriptor.detectAndCompute(main, None)
    keypoints_target, descriptor_target = descriptor.detectAndCompute(target, None)
    
    # 2. 패치 추출
    # 3. 벡터화 -> 비교
    # brute force == 알고리즘()
    # L1거리, L2거리
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptor_target, descriptor_main)
    
    # 가까운 정도(비슷한 정도)에 따라 정리함
    match = sorted(matches, key=lambda x : x.distance)
    
    # 시각화
    result = cv2.drawMatches(target, keypoints_target, main, keypoints_main, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS) 
    
    cv2.imshow(str(types), result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return ''
    
if __name__ == '__main__':
    # print(Descriptor.SIFT.name)
    # print(Descriptor.SIFT.value)
    
    # main 이미지 -> 매대 사진
    # target 이미지 -> 제품 사진
    main_path = './data/main03.jpg'
    target_path = './data/target03.jpg'
    
    main = cv2.imread(main_path, cv2.IMREAD_GRAYSCALE)
    target = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
    # types = 0
    # types = Descriptor.SIFT
    types = Descriptor.ORB.value
    
    matching(main, target, types)