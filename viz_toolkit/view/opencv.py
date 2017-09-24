import numpy as np
import cv2

class ImageView(object):
	""" A class for displaying opencv images using built in opencv functions

	"""
	def __init__(self, img=None, timestamp=0.0, windowName='img'):
		self._windowName = windowName
		self._rawImg = img
		self._imgToPlot = img
		self._timeStamp = timestamp
		self._frameCount = 0

		cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)

	def updateImage(self, img, timestamp=None):
		self._rawImg = img
		self._imgToPlot = np.copy(img)

		if timestamp is not None:
			self._timeStamp = timestamp

		self._frameCount += 1

	def clearImage(self):
		self._imgToPlot = self._rawImg

	def plotTracksColoredByScore(self, tracks):
		if len(tracks) < 1:
			return

		scores = [-t.score for t in tracks]
		maxScore = max(scores)

		colors = [(0, int((-t.score/maxScore)*255), 255 - int((-t.score/maxScore)*255))
					for t in tracks]

		self.plotTracks(tracks, colors)


	def plotTracks(self, tracks, colors=None):
		if self._imgToPlot is None:
			print('Please provide Image first')
			return 

		if colors is None:
			colors = [(255, 0, 0)] * len(tracks)

		for track, color in zip(tracks, colors):
			points = np.int32(np.asarray(track.positions).reshape(-1,2))
			cv2.polylines(self._imgToPlot, [points], True, color, thickness=2)

	def plot(self):
		if self._imgToPlot is None:
			print('Please provide Image first')
			return
		font = cv2.FONT_HERSHEY_SIMPLEX
		annotation = "Frame: " + str(self._frameCount)

		imgHeight, imgWidth = self._imgToPlot.shape[:2]
		cv2.putText(self._imgToPlot, annotation, (100, imgHeight-20), font, fontScale=3, 
					color=(255,255,255), thickness=5, lineType=cv2.LINE_AA)
		cv2.imshow(self._windowName, self._imgToPlot)
		cv2.waitKey(1)

	def save(self, filename):
		self.plot()
		cv2.imwrite(filename, self._imgToPlot)