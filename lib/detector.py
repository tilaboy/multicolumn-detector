import numpy as np
import cv2
#from pdf2image import convert_from_path, convert_from_bytes

class Detector():
    def __init__(self):
        self.letter_kernel = np.ones((2,1), np.uint8)
        self.word_kernel = np.ones((1,4), np.uint8)
        self.line_kernel_1 = np.ones((1,5), np.uint8)
        self.line_kernel_2 = np.ones((2,4), np.uint8)
        self.par_kernel = np.ones((5,5), np.uint8)
        self.margin_kernel = np.ones((20,5), np.uint8)


    def load_image(self, filename):
        image = cv2.imread(filename)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        (_, threshold) = cv2.threshold(
            image_gray,
            0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        return image, threshod

    @staticmethod
    def contour_and_draw(threshold, orig_image):
        image = orig_image.copy()
        (contours, _) = cv2.findContours(
            threshold,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in contours:
                (x, y, w, h) = cv2.boundingRect(cnt)
                cv2.rectangle(image, (x-1,y-5), (x+w, y+h), (0, 255, 0), 1)
        return image


    def find_letter(self, threshold, image):

        morph_image = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, self.letter_kernel, iterations=3)
        letter_image = cv2.erode(morph_image,self.letter_kernel,iterations=1)
        return self.contour_and_draw(letter_image, image)


    def find_word(self, threshold, image):
        morph_image = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, self.letter_kernel, iterations=2)
        word_image = cv2.dilate(morph_image, self.word_kernel, iterations=1)
        return self.contour_and_draw(word_image, image)


    def find_line(self, threshold, image):
        morph_image = cv2.morphologyEx(threshold,cv2.MORPH_CLOSE,self.line_kernel_2,iterations=2)
        line_image = cv2.dilate(morph_image,self.line_kernel_1,iterations=5)
        return self.contour_and_draw(line_image, image)


    def find_par(self, threshold, image):
        par_image = cv2.dilate(threshold,self.par_kernel,iterations=3)
        return self.contour_and_draw(par_image, image)


    def find_margin(self, threshold, image):
        margin_image = cv2.dilate(threshold,self.margin_kernel,iterations=5)
        return self.contour_and_draw(margin_image, image)


if __name__ == '__main__':
    file = 'foo.jpg'
    #images_from_path = convert_from_path('foo.pdf',output_folder=".", fmt='JPEG')
    detector = Detector()
    image, th = detector.load_image(file)
    output_letter = detector.find_letter(th,image)
    output_word = detector.find_word(th,image)
    output_line = detector.find_line(th,image)
    output_par = detector.find_par(th,image)
    output_margin = detector.find_margin(th,image)
    # special case for the 5th output because margin with paragraph is just the 4th output with margin
    cv2.imwrite("output_letter.jpg", output_letter)
    cv2.imwrite("output_word.jpg", output_word)
    cv2.imwrite("output_line.jpg", output_line)
    cv2.imwrite("output_par.jpg", output_par)
    cv2.imwrite("output_margin.jpg", output_margin)
