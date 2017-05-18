import numpy as np
import matplotlib.pyplot as plt
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


class OverlayView(object):
	""" A class for displaying opencv images in matplotlib

		Not for displaying images every frame, use only to create, display and save
		intermediate and final results
	"""

	def __init__(self, grid, img=None, timestamp=0.0):
		if img is not None:
			self._imgToPlot = np.copy(img[:,:,::-1])
		else:
			self._imgToPlot = None

		self._grid = grid

		self._fig = plt.figure(figsize=(10, 6), dpi=100)
		self._ax = self._fig.add_subplot(111)
		self._fig.subplots_adjust(right=0.85)
		self._cax = self._fig.add_axes([0.9, 0.15, 0.05, 0.7])


		self._cb = None
		self._colorbar = None
		self._img = None

		self._cmap = plt.cm.jet
		self._cmap._init()
		self._cmap._lut[:,-1] = np.linspace(0, 0.8, 255+4)

		self._timeStamp = timestamp
		self._frameCount = 0
		self._fig.set_size_inches(10, 6)


	def updateImage(self, img, timestamp=None):
		self._imgToPlot = np.copy(img[:,:,::-1])

		print(self._imgToPlot.shape)

		if self._img is None:
			self._img = self._ax.imshow(self._imgToPlot, origin='lower')
		else:
			extents = self._img.get_extent()
			self._img.set_data(self._imgToPlot)
			self._img.set_extent(extents)
			self._img.changed()

		if timestamp is not None:
			self._timeStamp = timestamp

		self._frameCount += 1
		self.plot()

	def clearColorBar(self):
		if self._colorbar is not None:
			self._colorbar.remove()
			self._colorbar = None

		self._cb = None

	def contourf(self, binnedScores, weighted=True):
		gridX, gridY = self._grid.mgrid

		if weighted:
			binFunc = lambda x,y: sum(binnedScores[(x,y)])
		else:
			binFunc = lambda x,y: len(binnedScores[(x,y)])

		vecFunc = np.vectorize(binFunc)

		values = vecFunc(gridX, gridY)

		self._cb = self._ax.contourf(gridX, gridY, values, 15, cmap=self._cmap)
		self._colorbar = plt.colorbar(self._cb, cax=self._cax)

	def hist2d(self, measurements, weighted=True):
		if len(measurements) < 1:
			return

		print(len(measurements))
		points = np.asarray([m.point for m in measurements])
		weights = None
		if weighted:
			weights = np.asarray([-m.score for m in measurements])

		_,_,_,newHist = self._ax.hist2d(points[:, 0], points[:, 1], weights=weights, 
			normed=True, bins=self._grid.edges, cmap=self._cmap)

		if self._cb is None:
			self._cb = newHist
			self._cb.changed()
		else:
			self._cb.set_data = newHist
			self._cb.changed()

		if self._colorbar is None:
			self._colorbar = self._fig.colorbar(self._cb, ax=self._ax, cax=self._cax, cmap=self._cmap)
			self._cb.changed()
		
		self._colorbar.set_clim(0, 1.0)
		self._colorbar.draw_all()
		self.plot()

		_,_,_,newHist = self._ax.hist2d(points[:, 0], points[:, 1], weights=weights, 
			normed=True, bins=self._grid.edges, cmap=self._cmap)
		self._cb.set_data = newHist
		self._cb.changed()
		self._colorbar.set_clim(0, 1.0)
		self._colorbar.draw_all()
		self.plot()


	def plot(self, pause=0.001):
		self._fig.canvas.draw()
		plt.show()
		plt.pause(pause)

	def save(self, filename):
		self._ax.set_title(f"Measurements at {self._timeStamp:.2f} seconds")
		self._fig.set_size_inches(10, 6)
		self._fig.savefig(filename, dpi=100, bbox_inches='tight')


class FieldOverlayView(object):
	""" View for plotting fields overlaid over images

	"""

	def __init__(self, grid, img=None, timestamp=0.0):
		if img is not None:
			self._imgToPlot = np.copy(img[:,:,::-1])
		else:
			self._imgToPlot = None

		self._grid = grid

		self._fig = plt.figure(figsize=(10, 6), dpi=100)
		self._ax = self._fig.add_subplot(111)
		self._fig.subplots_adjust(right=0.85)
		self._cax = self._fig.add_axes([0.9, 0.15, 0.05, 0.7])

		self._quiver = None
		self._cb = None
		self._colorbar = None
		self._img = None

		self._timeStamp = timestamp
		self._frameCount = 0
		self._fig.set_size_inches(10, 6)


	def updateImage(self, img, timestamp=None):
		self._imgToPlot = np.copy(img[:,:,::-1])

		print(self._imgToPlot.shape)

		if self._img is None:
			self._img = self._ax.imshow(self._imgToPlot, origin='lower')
		else:
			extents = self._img.get_extent()
			self._img.set_data(self._imgToPlot)
			self._img.set_extent(extents)
			self._img.changed()

		if timestamp is not None:
			self._timeStamp = timestamp

		self._frameCount += 1
		self._fig.canvas.draw()
		plt.pause(0.1)

	def drawField(self, field):
		xGrid, yGrid = self._grid.mgrid
		xSamples, ySamples = field.sampleAtGrid(xGrid, yGrid)
		magnitudes = np.sqrt(xSamples**2 + ySamples**2)

		self._clim = [0, magnitudes.max()]

		if (self._quiver is None):
			self._quiver = self._ax.quiver(xGrid, yGrid, xSamples, ySamples, magnitudes,
					clim=self._clim, angles='xy', scale_units='xy', scale=1, cmap=plt.cm.inferno)
		else:
			self._quiver.set_UVC(xSamples, ySamples, magnitudes)
			self._quiver.changed()


		#self._ax.text(85, 48, self._annotation)

		#self._ax.axis(self._field.plotExtents)

		#self._ax.minorticks_on()
		#self._ax.grid(which='both', alpha=1.0, linewidth=1)
		#self._ax.tick_params(which='both', direction='out')
		if self._colorbar is None:
			self._colorbar = self._fig.colorbar(self._quiver, ax=self._ax, cax=self._cax, cmap=plt.cm.inferno)

		self._colorbar.set_clim(*self._clim)
		self._colorbar.draw_all()
		self._fig.canvas.draw()
		self.plot()


	def plot(self, pause=0.001):
		self._fig.canvas.draw()
		plt.show()
		plt.pause(pause)

	def save(self, filename):
		self._ax.set_title(f"Field Approximation at {self._timeStamp:.2f} seconds")
		self._fig.set_size_inches(10, 6)
		self._fig.savefig(filename, dpi=100, bbox_inches='tight')
