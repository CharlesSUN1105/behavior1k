from b1k.pybullet_utils import init_client, load_r1pro, spin, print_joint_types


def main():
    client = init_client(gui=False)
    r1pro = load_r1pro(client)
    print_joint_types(client, r1pro)
    # spin(client)


if __name__ == "__main__":
    main()
