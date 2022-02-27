from bench import VkittiDataset, KittiDataset, TartanAirDataset

if __name__ == "__main__":

    # vk2 = VkittiDataset()
    # for i in range(len(vk2.data_dirs)):
    #      # Forward
    #      sdso_vk2_fwd = vk2.get_sdso(i, reverse=False)
    #      sdso_vk2_fwd.run()
    #      sdso_vk2_fwd.save()
    #      # Reverse
    #      sdso_vk2_rev = vk2.get_sdso(i, reverse=True)
    #      sdso_vk2_rev.run()
    #      sdso_vk2_rev.save()

    # kit = KittiDataset()
    # for i in range(len(kit.data_dirs)):
    #     # Forward
    #     sdso_kit_fwd = kit.get_sdso(i, reverse=False)
    #     sdso_kit_fwd.run()
    #     sdso_kit_fwd.save()
    #     # Reverse
    #     sdso_kit_rev = kit.get_sdso(i, reverse=True)
    #     sdso_kit_rev.run()
    #     sdso_kit_rev.save()

    tta = TartanAirDataset()
    # for i in range(len(tta.data_dirs)):
    #     sdso_tta_fwd = tta.get_sdso(i, reverse=False)
    #     sdso_tta_fwd.run()
    #     sdso_tta_fwd.save()
    #     sdso_tta_rev = tta.get_sdso(i, reverse=True)
    #     sdso_tta_rev.run()
    #     sdso_tta_rev.save()

    # for i in range(len(vk2.data_dirs)):
    #     dsol_vk2_fwd = vk2.get_dsol(i, reverse=False)
    #     dsol_vk2_fwd.run()
    #     dsol_vk2_rev = vk2.get_dsol(i, reverse=True)
    #     dsol_vk2_rev.run()

    # for i in range(len(kit.data_dirs)):
    #     dsol_kit_fwd = kit.get_dsol(i, reverse=False)
    #     dsol_kit_fwd.run()
    #     dsol_kit_rev = kit.get_dsol(i, reverse=True)
    #     dsol_kit_rev.run()

    for i in range(len(tta.data_dirs)):
        dsol_tta_fwd = tta.get_dsol(i, reverse=False)
        dsol_tta_fwd.run()
        dsol_tta_rev = tta.get_dsol(i, reverse=True)
        dsol_tta_rev.run()
