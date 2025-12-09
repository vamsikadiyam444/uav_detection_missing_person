# uav_detection_missing_person

# Overview

This Project provides a concise, practical test procedure for the UAV-based missing-person detection system. It covers three scenarios (Daylight, Low-Light, Windy), required equipment, acceptance criteria, test 

steps, logging, GPS,  and AWS SES notifications.
________________________________________
# Objectives

•	Validate end-to-end detection, identification, geolocation, and notification.

•	Verify system performance in: Daylight, Low-Light (IR/Thermal), and Windy (dual-sensor fusion).

•	Confirm Google Maps links and AWS SES notifications are produced for valid detections.
________________________________________
# Test Setup

Folders & Files - images/ — input images - missing_persons/ — reference images - output/ — annotated images - gps_log.csv — optional: filename,lat,lon,alt

Software/Environment - Python 3.12 (venv) - Packages: you can refer from requirements.txt

Hardware - UAV with RGB and IR/Thermal cameras (time-synced) - Ground control laptop/edge device - Optional: wind simulator (fans), lighting rigs

# Install required Python libraries:

```bash

pip install -r requirements.txt

```

# AWS SES Test Steps
Steps to Set Up SES  Emails from the sender and the recipient

 # Step 1: Log in to AWS SES

 Navigate to the AWS Management Console.

 Search for SES (Simple Email Service) and open it.

 SES resources are area-specific, therefore make sure you choose the right region (us-east-1, for example).

 # Step 2: Verify Sender Email

 The sender email is the one that will appear in the "From" field of your email.

 In the SES console, navigate to Verified identities → Create identity.

 Select Email address and click Next.

 Enter the email address you wish to send from, such as vamsikadiyam444@gmail.com.

 Click Create identity.

 Check the inbox of the sender email and click the verification link issued by AWS.

 Once validated, the sender email will get a next to it in SES.

 # Step 3: Use Sandbox Mode to Confirm Receiver Email

 When SES is in sandbox mode, you can only send emails to validated addresses.

 Verified identities → Create identity in the SES console.

 Select Email address and click Next.

 Create an identity and enter the recipient's email address, such as m.komal12345@gmail.com.

 Check the recipient email inbox and click the verification link.

 The recipient email is then prepared to accept emails from SES in sandbox mode.

________________________________________

# Set up AWS credentials for SES:

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

# Run Instructions

just run 

``base

python python2.py

```


