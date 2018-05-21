import numpy as np
import matplotlib.pyplot as plt

from primitives.grid import Grid

plt.ion()

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
		self._ax.set_axis_off()
		self._cax = None
		#self._fig.subplots_adjust(right=0.85)
		#self._cax = self._fig.add_axes([0.9, 0.15, 0.05, 0.7])


		self._cb = None
		self._colorbar = None
		self._img = None

		self._cmap = plt.cm.nipy_spectral
		self._cmap._init()
		self._cmap._lut[0:30,-1] = np.linspace(0, 1, 30)
		self._cmap._lut[30:,-1] = np.ones(225+4)#np.linspace(0, 1, 255+4)

		self._timeStamp = timestamp
		self._frameCount = 0
		self._fig.set_size_inches(10, 6)


	def updateImage(self, img, timestamp=None):
		self._imgToPlot = np.copy(img[:,:,::-1])

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

	def drawGrid(self, color='k'):
		xEdges, yEdges = self._grid.edges
		self._ax.vlines(xEdges, 0, max(yEdges), linewidths=0.5, color=color)
		self._ax.hlines(yEdges, 0, max(xEdges), linewidths=0.5, color=color)

	def createColorBar(self):
		self._fig.subplots_adjust(right=0.85)
		self._cax = self._fig.add_axes([0.9, 0.15, 0.05, 0.7])


	def clearColorBar(self):
		if self._colorbar is not None:
			self._colorbar.remove()
			self._colorbar = None

		self._cb = None

	def contourf(self, binnedScores, weighted=True):
		if self._cax is None:
			self.createColorBar()

		gridX, gridY = self._grid.mgrid

		if weighted:
			binFunc = lambda x,y: sum(binnedScores[self._grid.bin((x,y))]) / max(len(binnedScores[self._grid.bin((x,y))]), 1)
		else:
			binFunc = lambda x,y: len(binnedScores[self._grid.bin((x,y))])

		vecFunc = np.vectorize(binFunc)

		values = vecFunc(gridX, gridY)

		self._cb = self._ax.contourf(gridX, gridY, values, 25, cmap=self._cmap)
		self._colorbar = plt.colorbar(self._cb, cax=self._cax)

	def hist2d(self, measurements, weighted=True):
		if len(measurements) < 1:
			return

		if self._cax is None:
			self.createColorBar()

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

	def plotTracks(self, tracks, colors=None, legend=None, labelled=False):
		if self._imgToPlot is None:
			print('Please provide image first')
			return 

		if colors is None:
			colors = [(1.0, 0.0, 0.0)] * len(tracks)

		lines = []
		for track, color in zip(tracks, colors):
			points = np.int32(np.asarray(track.positions).reshape(-1,2))
			line = self._ax.plot(points[:,0], points[:,1], color=color, linewidth=1.0)
			lines.append(line[0])

		if legend is not None:
			plt.legend(lines, legend)

		if labelled:
			for track, color in zip(tracks, colors):
				points = np.int32(np.asarray(track.positions).reshape(-1,2))
				line = self._ax.plot(points[:,0], points[:,1], color=color, linewidth=1.0)
				text = self._ax.annotate(f"{track.id}", xy=(points[-1,0], points[-1,1]), xytext=(0,0),
									 textcoords='offset points', size=2, color='blue',
									 horizontalalignment='center', verticalalignment='bottom')

	def plotDirectedTracks(self, tracks):
		if self._imgToPlot is None:
			print('Please provide image first')
			return

		for track in tracks:
			_, endPoint = track.getLastObservation()
			_, startPoint = track.getFirstObservation()

			vec = endPoint - startPoint
			vec = vec / np.linalg.norm(vec)
			xVal = vec[0]
			yVal = vec[1]

			greenVal = 0.5*yVal + 0.5
			redVal = -0.5*yVal + 0.5
			blueVal = 0.5*xVal + 0.5
			points = np.int32(np.asarray(track.positions).reshape(-1,2))
			line = self._ax.plot(points[:,0], points[:,1], color=(redVal, greenVal, 0.0))
			self._ax.scatter([endPoint[0]], [endPoint[1]], s=4., marker='o', c='white')

		self.plot()

	def plotAnnotatedLine(self, x, y):
		line = self._ax.plot(x, y, color='blue')
		midX = (x[0] + x[1]) / 2.0
		midY = (y[0] + y[1]) / 2.0

		xDiff = x[1] - x[0]
		yDiff = y[1] - y[0]

		magnitude = np.linalg.norm([xDiff, yDiff])

		text = self._ax.annotate(f"{magnitude:.2f}", xy=(np.max(x), np.max(y)), xytext=(0,-1.0),
								 textcoords='offset points', size=6, color='blue',
								 horizontalalignment='center', verticalalignment='bottom')

		sp1 = self._ax.transData.transform_point((x[0], y[0]))
		sp2 = self._ax.transData.transform_point((x[1], y[1]))

		rise = (sp2[1] - sp1[1])
		run = (sp2[0] - sp1[0])

		slope_degrees = np.degrees(np.arctan2(rise, run))
		#text.set_rotation(slope_degrees)


	def clearTracks(self):
		if self._imgToPlot is None:
			print('Please provide Image first')
			return

		self._img = None
		self._ax.clear()
		self.plot()

	def plot(self, pause=0.001):
		self._fig.canvas.draw()
		plt.tick_params(axis='both', which='both', bottom='off', top='off', labelbottom='off',
						right='off', left='off', labelleft='off')

		imgHeight, imgWidth = self._imgToPlot.shape[:2]
		self._ax.set_xlim([0,imgWidth])
		self._ax.set_ylim([0,imgHeight])
		plt.show()
		plt.pause(pause)

	def setTitle(self, title):
		self._ax.set_title(title)

	def save(self, filename):
		self._fig.set_size_inches(10, 6)
		self._fig.savefig(filename, dpi=300, bbox_inches='tight')

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
		self._ax.set_xticks([])
		self._ax.set_yticks([])
		self._fig.subplots_adjust(right=0.87)
		self._cax = self._fig.add_axes([0.9, 0.16, 0.05, 0.67])

		self._quiver = None
		self._cb = None
		self._colorbar = None
		self._img = None

		self._cmap = plt.cm.nipy_spectral
		self._cmap._init()
		self._cmap._lut[:,-1] = np.linspace(0, 0.8, 255+4)

		self._timeStamp = timestamp
		self._frameCount = 0
		self._fig.set_size_inches(10, 6)


	def updateImage(self, img, timestamp=None):
		self._imgToPlot = np.copy(img[:,:,::-1])

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

	def drawField(self, field, mask=None):
		self._cmap = plt.cm.nipy_spectral
		self._cmap._init()

		self._field = field
		xGrid, yGrid = self._grid.mgrid
		xSamples, ySamples = field.sampleAtGrid(xGrid, yGrid)
		magnitudes = np.sqrt(xSamples**2 + ySamples**2)

		if mask is not None:
			m = mask[yGrid.astype(int), xGrid.astype(int)]
			xSamples = np.ma.masked_array(xSamples, mask=m)
			ySamples = np.ma.masked_array(ySamples, mask=m)

		self._clim = [magnitudes.min(), magnitudes.max()]

		if (self._quiver is None):
			self._quiver = self._ax.quiver(xGrid, yGrid, xSamples, ySamples, magnitudes,
					clim=self._clim, cmap=plt.cm.nipy_spectral, angles='xy', scale_units='xy', scale=0.35)

			"""
			self._quiver = self._ax.quiver(xGrid, yGrid, xSamples, ySamples, magnitudes,
					clim=self._clim, angles='xy', scale_units='xy', scale=1, cmap=plt.cm.nipy_spectral,
					headwidth=2, headlength=4)
			"""

		else:
			self._quiver.set_UVC(xSamples, ySamples, magnitudes)
			self._quiver.changed()


		#self._ax.text(85, 48, self._annotation)

		#self._ax.axis(self._field.plotExtents)

		#self._ax.minorticks_on()
		#self._ax.grid(which='both', alpha=1.0, linewidth=1)
		#self._ax.tick_params(which='both', direction='out')
		if self._colorbar is None:
			self._colorbar = self._fig.colorbar(self._quiver, ax=self._ax, cax=self._cax, cmap=plt.cm.nipy_spectral)

		self._colorbar.set_clim(*self._clim)
		#self._colorbar.set_label('px/s')
		self._colorbar.draw_all()
		self._fig.canvas.draw()
		self.plot()

	def clearField(self):
		if (self._quiver is not None):
			self._quiver.remove()

	def clearAxes(self):
		self._img = None
		self._ax.clear()
		self._quiver = None

	def drawVariance(self, axis=0, grid=None):
		if (self._field is None):
			print("Error: Field not set")
			return

		self._cmap = plt.cm.nipy_spectral
		self._cmap._init()
		self._cmap._lut[:,-1] = np.linspace(0, 0.8, 255+4)

		if grid is None:
			xGrid, yGrid = self._grid.mgrid
		else:
			xGrid, yGrid = grid.mgrid
		
		variance = self._field.sampleVarAtGrid(xGrid, yGrid)

		var = variance[axis]

		self._cb = self._ax.contourf(xGrid, yGrid, var, 15, cmap=self._cmap)
		self._colorbar = plt.colorbar(self._cb, cax=self._cax)
		self._colorbar.draw_all()
		self.plot()

	def plot(self, pause=0.001):
		self._fig.canvas.draw()
		plt.show()
		plt.pause(pause)

	def setTitle(self, title):
		self._ax.set_title(title)

	def setField(self, field):
		self._field = field

	def setGrid(self, grid):
		self._grid = grid

	def save(self, filename):
		self._fig.set_size_inches(10, 6)
		self._fig.savefig(filename, dpi=300, bbox_inches='tight')

class FieldView(object):
	""" View for plotting fields

	"""

	def __init__(self, field=None, grid=None):
		self._field = field

		if (field is not None and grid is None):
			self._grid = Grid.from_bounds(field.extents.bounds, cellSize=(60,50))
		else:
			self._grid = grid

		self._fig = plt.figure(figsize=(10, 6), dpi=100)
		self._ax = self._fig.add_subplot(111)
		self._cax = None

		self._quiver = None
		self._cb = None
		self._colorbar = None

		self._cmap = plt.cm.nipy_spectral

		self._frameCount = 0
		self._fig.set_size_inches(10, 6)

	def clearColorBar(self):
		if self._colorbar is not None:
			self._colorbar.remove()
			self._colorbar = None

		self._cb = None

	def clearPlot(self):
		self._ax.clear()
		self._quiver = None

	def drawField(self):
		if (self._field is None):
			print("Error: Field not set")
			return

		if self._cax is None:
			self._fig.subplots_adjust(right=0.85)
			self._cax = self._fig.add_axes([0.9, 0.15, 0.05, 0.7])

		xGrid, yGrid = self._grid.mgrid
		xSamples, ySamples = self._field.sampleAtGrid(xGrid, yGrid)
		magnitudes = np.sqrt(xSamples**2 + ySamples**2)

		self._clim = [0, magnitudes.max()]

		if (self._quiver is None):
			self._quiver = self._ax.quiver(xGrid, yGrid, xSamples, ySamples, magnitudes,
					clim=self._clim, angles='xy', scale_units='xy', scale=1, cmap=plt.cm.nipy_spectral)
		else:
			self._quiver.set_UVC(xSamples, ySamples, magnitudes)
			self._quiver.changed()


		#self._ax.text(85, 48, self._annotation)

		#self._ax.axis(self._field.plotExtents)

		#self._ax.minorticks_on()
		#self._ax.grid(which='both', alpha=1.0, linewidth=1)
		#self._ax.tick_params(which='both', direction='out')
		if self._colorbar is None:
			self._colorbar = self._fig.colorbar(self._quiver, ax=self._ax, cax=self._cax, cmap=plt.cm.nipy_spectral)

		self._colorbar.set_clim(*self._clim)
		self._colorbar.draw_all()
		self._fig.canvas.draw()
		self.plot()

	def drawScalarField(self):
		if (self._field is None):
			print("Error: Field not set")
			return

		xGrid, yGrid = self._grid.mgrid
		values = self._field.sampleAtGrid(xGrid, yGrid)

		self._cb = self._ax.contourf(xGrid, yGrid, values, 15, cmap=self._cmap)
		self._colorbar = plt.colorbar(self._cb, cax=self._cax)
		self._colorbar.draw_all()
		self.plot()

	def drawMeasurementLocations(self, measurements):
		points = np.array([list(m.point) for m in measurements])
		samples = self._ax.scatter(points[:,0], points[:,1], c='black', label='Sampling Locations')
		self._ax.legend(loc='upper right')
		self._ax.axis(self._field.plotExtents)
		self.plot()

	def drawMeasurementContours(self, measurements):
		pass

	def plotTracks(self, tracks, colors=None, legend=None, labelled=False):
		if colors is None:
			colors = [(0.0, 0.0, 0.0)] * len(tracks)
		elif len(colors) == 1:
			colors = colors * len(tracks)

		lines = []
		for track, color in zip(tracks, colors):
			points = np.asarray(track.positions).reshape(-1,2)
			line = self._ax.plot(points[:,0], points[:,1], color=color, linewidth=2.0)
			lines.append(line[0])

		if legend is not None:
			plt.legend(lines, legend)

		if labelled:
			for track, color in zip(tracks, colors):
				points = np.int32(np.asarray(track.positions).reshape(-1,2))
				line = self._ax.plot(points[:,0], points[:,1], color=color, linewidth=1.0)
				text = self._ax.annotate(f"{track.id}", xy=(points[-1,0], points[-1,1]), xytext=(0,0),
									 textcoords='offset points', size=2, color='blue',
									 horizontalalignment='center', verticalalignment='bottom')

	def plotAnnotatedLine(self, x, y):
		line = self._ax.plot(x, y, color='blue')
		midX = (x[0] + x[1]) / 2.0
		midY = (y[0] + y[1]) / 2.0

		xDiff = x[1] - x[0]
		yDiff = y[1] - y[0]

		magnitude = np.linalg.norm([xDiff, yDiff])

		text = self._ax.annotate(f"{magnitude:.2f}", xy=(np.max(x), np.max(y)), xytext=(0,-1.0),
								 textcoords='offset points', size=6, color='blue',
								 horizontalalignment='center', verticalalignment='bottom')

		sp1 = self._ax.transData.transform_point((x[0], y[0]))
		sp2 = self._ax.transData.transform_point((x[1], y[1]))

		rise = (sp2[1] - sp1[1])
		run = (sp2[0] - sp1[0])

		slope_degrees = np.degrees(np.arctan2(rise, run))

	def drawVariance(self, axis=0):
		if (self._field is None):
			print("Error: Field not set")
			return

		xGrid, yGrid = self._grid.mgrid
		variance = self._field.sampleVarAtGrid(xGrid, yGrid)

		var = variance[axis]

		self._cb = self._ax.contourf(xGrid, yGrid, var, 15, cmap=self._cmap)
		self._colorbar = plt.colorbar(self._cb, cax=self._cax)
		self._colorbar.draw_all()
		self.plot()


	def plot(self, pause=0.001):
		self._fig.canvas.draw()
		plt.show()
		plt.pause(pause)

	def setTitle(self, title):
		self._ax.set_title(title)

	def setField(self, field):
		self._field = field

	def setGrid(self, grid):
		self._grid = grid

	def save(self, filename):
		self._fig.set_size_inches(10, 6)
		self._fig.savefig(filename, dpi=300, bbox_inches='tight')
