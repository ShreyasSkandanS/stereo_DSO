from bench import VkittiDataset, KittiDataset, TartanAirDataset

if __name__ == "__main__":
    vk2 = VkittiDataset()
    #vk2.write_all_gt()
    vk2.write_single_gt(0)

    #kit = KittiDataset()
    #kit.write_all_gt()

    # tta = TartanAirDataset()
    # tta.write_all_gt()
