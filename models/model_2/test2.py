from RoadNetEnv import raw_env
from NguyenNetwork import nguyenNetwork, traffic

try:
    test = raw_env()
    # Further actions with the test object
except Exception as e:
    print(f"An error occurred: {e}")