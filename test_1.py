import stainNorm_Macenko
import cv2

sampleImagePath = '/Users/jiangxiaofeng/Desktop/github_test/macenko_norm/Ref.png'

target = cv2.imread(sampleImagePath)
target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

normalizer = stainNorm_Macenko.Normalizer()
normalizer.fit(target)

test_img = '/Users/jiangxiaofeng/Desktop/macenko_test/raw.jpeg'

img = cv2.imread(test_img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
nor_img = normalizer.transform(img)
cv2.imwrite('/Users/jiangxiaofeng/Desktop/macenko_test/new111.jpeg', cv2.cvtColor(nor_img, cv2.COLOR_RGB2BGR))