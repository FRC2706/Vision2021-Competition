import zipfile

FILENAME = "visionCompPi20.zip"
FILENAME2 = "visionCompPi21.zip"
FILENAME3 = "visionPracticePi20.zip"
FILENAME4 = "visionPracticePi21.zip"
FILENAME5 = "visionCompPi22.zip"
FILENAME6 = "visionPracticePi22.zip"

#create a ZipFile object
zipObj = zipfile.ZipFile(FILENAME, 'w')

# Add module files to the zip
zipObj.write('ControlPanel.py')
zipObj.write('DistanceFunctions.py')
zipObj.write('FindBall.py')
zipObj.write('FindCone.py')
zipObj.write('FindTarget.py')
zipObj.write('FindStaticElement.py')
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
zipObj2.write('FindCone.py')
zipObj2.write('FindTarget.py')
zipObj2.write('FindStaticElement.py')
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
zipObj3.write('FindCone.py')
zipObj3.write('FindTarget.py')
zipObj3.write('FindStaticElement.py')
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
zipObj4.write('FindCone.py')
zipObj4.write('FindTarget.py')
zipObj4.write('FindStaticElement.py')
zipObj4.write('VisionConstants.py')
zipObj4.write('VisionMasking.py')
zipObj4.write('VisionUtilities.py')
zipObj4.write('NetworkTablePublisher.py')
zipObj4.write('MergeFRCPipeline.py','uploaded.py')
zipObj4.write('CornersVisual4.py')
zipObj4.write('pipelineConfigPractPi21.json', 'pipelineConfig.json')

zipObj5 = zipfile.ZipFile(FILENAME5, 'w')

zipObj5.write('ControlPanel.py')
zipObj5.write('DistanceFunctions.py')
zipObj5.write('FindBall.py')
zipObj5.write('FindTarget.py')
zipObj5.write('FindStaticElement.py')
zipObj5.write('VisionConstants.py')
zipObj5.write('VisionMasking.py')
zipObj5.write('VisionUtilities.py')
zipObj5.write('NetworkTablePublisher.py')
zipObj5.write('MergeFRCPipeline.py','uploaded.py')
zipObj5.write('CornersVisual4.py')
zipObj5.write('pipelineConfigPi22.json', 'pipelineConfig.json')


zipObj6 = zipfile.ZipFile(FILENAME6, 'w')

zipObj6.write('ControlPanel.py')
zipObj6.write('DistanceFunctions.py')
zipObj6.write('FindBall.py')
zipObj6.write('FindTarget.py')
zipObj6.write('FindStaticElement.py')
zipObj6.write('VisionConstants.py')
zipObj6.write('VisionMasking.py')
zipObj6.write('VisionUtilities.py')
zipObj6.write('NetworkTablePublisher.py')
zipObj6.write('MergeFRCPipeline.py','uploaded.py')
zipObj6.write('CornersVisual4.py')
zipObj6.write('pipelineConfigPractPi22.json', 'pipelineConfig.json')

print("I have written the files: \n" + FILENAME + ", \n" + FILENAME2 + ", \n" + FILENAME3 + ", \n" + FILENAME4 + ", \n" + FILENAME5 + " and \n" + FILENAME6 + "\n")