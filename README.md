# uav_detection_missing_person

# Overview

This Project provides a concise, practical test procedure for the UAV-based missing-person detection system. It covers three scenarios (Daylight, Low-Light, Windy), required equipment, acceptance criteria, test 

steps, logging, GPS,  and AWS SNS notifications.
________________________________________
# Objectives

•	Validate end-to-end detection, identification, geolocation, and notification.

•	Verify system performance in: Daylight, Low-Light (IR/Thermal), and Windy (dual-sensor fusion).

•	Confirm Google Maps links and AWS SNS notifications are produced for valid detections.
________________________________________
# Test Setup

Folders & Files - images/ — input images - missing_persons/ — reference images - output/ — annotated images - gps_log.csv — optional: filename,lat,lon,alt

Software/Environment - Python 3.12 (venv) - Packages: you can refer from requirements.txt

Hardware - UAV with RGB and IR/Thermal cameras (time-synced) - Ground control laptop/edge device - Optional: wind simulator (fans), lighting rigs

# Install required Python libraries:

```bash

pip install -r requirements.txt

```

# AWS SNS Test Steps
1.	Create SNS topic and subscribe an email/SMS.
   
2.	Add AWS credentials to environment (or configure profile).

3.	Run detection; script publishes when match == True:

4.Subject: Missing person detected: <NAME>

5.	Message: includes name, confidence, Google Maps link, image path, timestamp

6.	Verify receipt and content formatting.
________________________________________

# Set up AWS credentials for SNS:

```bash

export AWS_ACCESS_KEY_ID='YOUR_ACCESS_KEY'
export AWS_SECRET_ACCESS_KEY='YOUR_SECRET_KEY'
export AWS_DEFAULT_REGION='YOUR_REGION'

```

# Scenario A: Daylight Search (RGB)
1.	Capture geotagged RGB sweeps.

2.	Place subjects at known coordinates.

3.	Transfer images to images/.

4.	Run:
 
   python2.py --mode daylight

5.	Verify annotated images in output/ and printed Google Maps links.

6.	Confirm SNS notification receipt.

Record: detections, matches, GPS, processing time.

# Scenario B: Low-Light Search (IR/Thermal fallback)

1.	Capture synchronized RGB (low-light) + IR frames.

2.	Transfer images to images/ (IR frames in ir_images/ if used).

3.	Run:

python2.py --mode lowlight

5.	Verify detections in output/; check IR fallback activation logs.

6.	Confirm SNS messages for matches.

Record: IR vs RGB detection comparison, false alarms.

# Scenario C: Windy Deployment (Dual-sensor fusion)

1.	Capture jittery frames with UAV/gimbal under gusts.

2.	Transfer to images/.

3.	Run:

python2.py --mode dualsensor

5.	Verify frame-stability filtering and fusion use; check annotated images.

6.	Confirm SNS message delivery and log processing times.

Record: skipped frames %, detection latency.


