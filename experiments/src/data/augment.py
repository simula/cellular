import albumentations as albu


def build_augmentations( augmentation_list, return_compose=True ):
    augmentations = [ ]
    for augmentation_config in augmentation_list:
        if augmentation_config[ "name" ] == "OneOf":
            augmentation = albu.OneOf(
                build_augmentations( augmentation_config[ "augmentations" ], False ),
                **augmentation_config[ "params" ])
        else:
            augmentation = getattr(albu, augmentation_config[ "name" ]
                )( **augmentation_config[ "params" ] )
        augmentations.append( augmentation )
    if return_compose:
        return albu.Compose( augmentations )
    return augmentations