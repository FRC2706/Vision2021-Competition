import zipfile

FILENAME = "visionCompPi20.zip"
FILENAME2 = "visionCompPi21.zip"
FILENAME3 = "visionPracticePi20.zip"
FILENAME4 = "visionPracticePi21.zip"

#create a ZipFile object
zipObj = zipfile.ZipFile(FILENAME, 'w')

# Add module files to the zip
zipObj.write('ControlPanel.py')
zipObj.write('DistanceFunctions.py')
zipObj.write('FindBall.py')
zipObj.write('FindTarget.py')
zipObj.write('VisionConstants.py')
zipObj.write('VisionMasking.py')
zipObj.write('VisionUtilities.py')
zipObj.write('NetworkTablePublisher.py')
zipObj.write('MergeFRCPipeline.py','uploaded.py')
zipObj.write('CornersVisual4.py')
zipObj.write('pipelineConfigPi20.json', 'pipelineConfig.json')


zipObj2 = zipfile.ZipFile(FILENAME2, 'w')

zipObj2.write('ControlPanel.py')
zipObj2.write('DistanceFunctions.py')
zipObj2.write('FindBall.py')
zipObj2.write('FindTarget.py')
zipObj2.write('VisionConstants.py')
zipObj2.write('VisionMasking.py')
zipObj2.write('VisionUtilities.py')
zipObj2.write('NetworkTablePublisher.py')
zipObj2.write('MergeFRCPipeline.py','uploaded.py')
zipObj2.write('CornersVisual4.py')
zipObj2.write('pipelineConfigPi21.json', 'pipelineConfig.json')

zipObj3 = zipfile.ZipFile(FILENAME3, 'w')

zipObj3.write('ControlPanel.py')
zipObj3.write('DistanceFunctions.py')
zipObj3.write('FindBall.py')
zipObj3.write('FindTarget.py')
zipObj3.write('VisionConstants.py')
zipObj3.write('VisionMasking.py')
zipObj3.write('VisionUtilities.py')
zipObj3.write('NetworkTablePublisher.py')
zipObj3.write('MergeFRCPipeline.py','uploaded.py')
zipObj3.write('CornersVisual4.py')
zipObj3.write('pipelineConfigPractPi20.json', 'pipelineConfig.json')

zipObj4 = zipfile.ZipFile(FILENAME4, 'w')

zipObj4.write('ControlPanel.py')
zipObj4.write('DistanceFunctions.py')
zipObj4.write('FindBall.py')
zipObj4.write('FindTarget.py')
zipObj4.write('VisionConstants.py')
zipObj4.write('VisionMasking.py')
zipObj4.write('VisionUtilities.py')
zipObj4.write('NetworkTablePublisher.py')
zipObj4.write('MergeFRCPipeline.py','uploaded.py')
zipObj4.write('CornersVisual4.py')
zipObj4.write('pipelineConfigPractPi21.json', 'pipelineConfig.json')

print("I have wrote the file: " + FILENAME + ", " + FILENAME2 + ", " + FILENAME3 + " and " + FILENAME4)