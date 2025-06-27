- **Update 11.06.**
With a new approach and further testing the position issue was solved.
- new issue: p-value do not match (see next section) FIXED

- **Update 22.06.** 
- Fixed p-value issue
- Noted that the statplot of shape_dd is shifted by window size (x2?) FIXED
- deriving from that the marked drifts are mismatching by shift FIXED
- the last position of dataset is detected as drift by shape_dd FIXED

- **Update 27.06.**
- fixed last position issue in shape_dd
- shape online now gets all data from the update function
- when a possible drift is at a position less than 2* window size, drift will not be detected.
- significantly better readme structure