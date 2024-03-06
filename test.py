# from pypylon import pylon
# import cv2
# import time

# camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
# camera.Open()

# camera.TriggerSource.SetValue("Line3")
# camera.TriggerMode.SetValue("On")

# im_width = camera.Width.GetValue()
# im_height = camera.Height.GetValue()

# camera.Close()

# camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

# t0 = time.perf_counter()
# while True:
#     grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
#     if grabResult.GrabSucceeded():
        
#         cv2.imshow("Frame", cv2.resize(grabResult.GetArray(), (im_width//2, im_height//2)))
#         if cv2.waitKey(1) == ord("q"): break
#         td = time.perf_counter() - t0
#         print(f"{1/td:.4f}")
#         t0 = time.perf_counter()
        
# cv2.destroyAllWindows()
# camera.StopGrabbing()




# from pathlib import Path
# import os

# dir = Path("test_dir/").joinpath("tmp")
# if not dir.exists():
#     dir.mkdir(parents=True, exist_ok=True)
# with open("test_dir/tmp/test.txt", 'w') as f:
#     f.write("test")
# for f in dir.glob("*"):
#     os.remove(f)
# dir.rmdir()




# import numpy as np
# import cv2

# blank = np.zeros((360, 480), dtype=np.uint8)
# txt = "waiting..."
# while True:

#     frame = cv2.putText(blank, txt, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
#     cv2.imshow("frame", frame)
#     res = cv2.waitKey(16)
#     if res == ord('q'):
#         break
#     elif res == ord('w'):
#         txt = "nuh~uh"
#     elif res == ord('e'):
#         txt = "E"
#     blank = np.zeros((360, 480), dtype=np.uint8)
# cv2.destroyAllWindows()
