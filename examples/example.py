import cv2
import facealignment


# Read sample images
single_face = cv2.imread("tests/samples/single_face.png")
multi_face = cv2.imread("tests/samples/multi_face.png")

# Instantiate FaceAlignmentTools class
tool = facealignment.FaceAlignmentTools()

# Align image with single face
aligned_img = tool.align(single_face)
screen_img = cv2.hconcat([single_face, aligned_img])
cv2.imshow("Aligned Example Image", screen_img)
cv2.waitKey(0)

# Align image with multiple faces
aligned_imgs = tool.align(multi_face, allow_multiface=True)
screen_img = cv2.hconcat([multi_face] + aligned_imgs)
cv2.imshow("Aligned Example Image", screen_img)
cv2.waitKey(0)
