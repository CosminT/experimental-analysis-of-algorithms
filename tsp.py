import numpy as np


class TspDrone:
    def __init__(self, truck_dist, drone_dist):
        self.truck_dist = truck_dist
        self.drone_dist = drone_dist

    def get_score(self, x):
        x = x.astype(int)
        p = [0]
        for el in range(x.shape[1]):
            if x[1][el] == 1 and x[1][el - 1] == 0:
                p.append(el)
        aux_p = p[1:].copy()
        aux_p.append(0)

        result_drum = []
        result_drum_type = []
        for el in range(len(p) - 1):
            drum = []
            drum_type = []
            drone = True
            data = x[0][p[el] + 1 : aux_p[el]]
            data_type = x[1][p[el] + 1 : aux_p[el]]
            from_d = x[0][p[el]]
            from_t = x[0][p[el]]

            for d in range(len(data_type)):
                if drone:
                    if data_type[d] == 1:
                        # drone = True
                        drum.append([from_d, data[d]])
                        drum.append([data[d], from_d])
                        drum_type.append(1)
                    else:
                        drone = False
                        drum.append([from_d, data[d]])
                        from_d = data[d]
                        drum_type.append(1)

                else:
                    drum.append([from_t, data[d]])
                    drum_type.append(0)
                    from_t = data[d]

            drum.append([from_d, x[0][aux_p[el]]])
            drum.append([from_t, x[0][aux_p[el]]])
            drum_type.append(1)
            drum_type.append(0)
            result_drum.append(drum)
            result_drum_type.append(drum_type)

        dist = 0

        for el in range(len(result_drum)):
            x = np.array(result_drum[el])
            y = np.array(result_drum_type[el])
            idx_dr = np.where(y == 1)
            idx_tr = np.where(y == 0)
            tour_dr = x[idx_dr]
            tour_tr = x[idx_tr]
            dist += max(
                self.truck_dist[tour_tr[:-1], tour_tr[1:]].sum(),
                self.drone_dist[tour_dr[:-1], tour_dr[1:]].sum()
            )

        return dist
