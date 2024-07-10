import gymnasium as gym
import numpy as np
import cv2




env = gym.make('FetchReachDense-v2', render_mode="rgb_array")


env.reset()
for i in range(1000):
    obs, reward, term, trun, info = env.step(np.random.random(size=4))
    rgb = env.render()
    print(i, end="\r")

    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    cv2.imshow("Curr", rgb)
    cv2.waitKey(30)
    # cv2.waitKey(1000)


    if term or trun:
        env.reset()

# End loop
cv2.destroyAllWindows()