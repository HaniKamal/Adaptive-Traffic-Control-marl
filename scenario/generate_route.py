import math
import random


def scenario():
    random.seed(42)  # make tests reproducible
    # demand per second from different directions

    with open("amman.rou.xml", "w") as routes:
        print("""<routes>
        <vType id="type_0" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="15.00" guiShape="passenger" emissionClass="HBEFA3/PC"/>
        <route edges="E0 E9" color="yellow" id="route_0"/>
        <route edges="-E9 -E0" color="yellow" id="route_1"/>
        <route edges="-E1 -E3 E8" color="yellow" id="route_2"/>
        <route edges="-E11 E1" color="yellow" id="route_3"/>
        <route edges="-E8 E3 E11" color="yellow" id="route_4"/>
        <route edges="-E11 -E3 E8" color="yellow" id="route_5"/>""", file=routes)

        vehNr = 0
        spawn_time = [0 for _ in range(6)]
        time = 0
        while time < 3600:

            for i in range(len(spawn_time)):
                if spawn_time[i] == time:
                    p = random.random()
                    inter_arrival_time = -math.log(1.0 - p) * 30
                    print('    <vehicle id="vehicle_%i" type="type_0" route="route_%i" depart="%i" color="0,1,0"/>' % (
                        vehNr, i, time), file=routes)
                    spawn_time[i] += max(int(inter_arrival_time), 1)
                    vehNr += 1

            time += 1

        print("</routes>", file=routes)


if __name__ == "__main__":
    scenario()
