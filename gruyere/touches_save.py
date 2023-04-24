from itertools import compress, product

from gruyere.design import Design
from gruyere.ops import convolve_pixel, get_convolved_idx
from gruyere.misc import from_list_id_couple_to_2tuples_ids
from gruyere.states import TouchState, PixelState, DesignState, FreeState


def add_solid_touches(des: Design, idxs: list[int, int], brush):
    for idx in idxs:
        # des = add_solid_touch(des, idx)
        des = add_solid_touch(des, idx, brush)

    return des


def add_void_touches(des: Design, idxs: list[int, int], brush):
    for idx in idxs:
        des = add_solid_touch(des._invert(), idx, brush)

    return des._invert()


# def add_solid_touch(des: Design, idx: int):
#     # Step 1 - Update the new VOID touch
#     t_solid = des.t_s.at[idx].set(TouchState.EXISTING)
#     return update_solid(des._replace(t_s=t_solid))

def add_solid_touch(des: Design, idx_touch: int, brush):
    idx_full_array = list(product(*map(range, des.x.shape)))
    idx_t_dilated = get_convolved_idx(des.x, idx_touch, brush)

    # No need to use the equation (B.5)
    # --> By default, all touches are considered valid

    # Step 1 - Update the design
    new_x = des.x.at[
        from_list_id_couple_to_2tuples_ids(idx_t_dilated)
    ].set(DesignState.SOLID)
    # Step 1bis - Update the new SOLID touch
    t_solid = des.t_s.at[idx_touch].set(TouchState.EXISTING)

    # Step 2 - Update associated SOLID pixels (B.3)
    # p_s_existing = D(t_s, b)
    p_solid = des.p_s.at[
        from_list_id_couple_to_2tuples_ids(idx_t_dilated)
    ].set(PixelState.EXISTING)
    # If pixels are already solid, they cannot be void
    p_void = des.p_v.at[
        from_list_id_couple_to_2tuples_ids(idx_t_dilated)
    ].set(PixelState.IMPOSSIBLE)

    # Step 3 - Update VOID touches
    # --> Where a solid touch exist, impossible to have a void touch (B.4)
    # t_v_impossible = D(p_s_existing, b)
    idx_mapped = list(map(
        lambda idx_var: get_convolved_idx(des.x, idx_var, brush),
        idx_t_dilated
    ))
    idx_t_double_dilated = list(set().union(*idx_mapped))
    t_void = des.t_v.at[
        from_list_id_couple_to_2tuples_ids(idx_t_double_dilated)
    ].set(TouchState.INVALID)

    # Step 4 - Find valid SOLID touches
    idx_valid = list(compress(
        idx_full_array,

        ~(
            (t_solid == TouchState.INVALID) | (t_solid == TouchState.EXISTING)
        ).flatten()
    ))

    # Step 5 - Find impossible SOLID pixels
    # # WORK HERE
    # idx_impossible = list(set(idx_t_dilated).union(
    #     *map(
    #         lambda idx_var: set(get_convolved_idx(des.x, idx_var, brush)),
    #         idx_valid
    #     )
    # ))
    # p_void.at[
    #     from_list_id_couple_to_2tuples_ids(idx_impossible)
    # ].set(PixelState.IMPOSSIBLE)


    # Step 6 - Find required SOLID pixels
    idx_required = list(compress(
        idx_full_array,
        # ~(
        #     (p_solid == PixelState.EXISING) | (p_void == PixelState.POSSIBLE)
        # ).flatten()
        (
            (p_solid == PixelState.POSSIBLE) & (p_void == PixelState.IMPOSSIBLE)
        ).flatten()
    ))

    # If at[[]] -> fulfil all the table
    if len(idx_required) > 0:
        p_solid = p_solid.at[
            from_list_id_couple_to_2tuples_ids(idx_required)
        ].set(PixelState.REQUIRED)

    # Step 7 - Find resolving SOLID touches
    # If at[[]] -> fulfil all the table
    if len(idx_required) > 0:
        idx_resolving = set(idx_valid).intersection(
            *map(
                lambda idx_var: set(get_convolved_idx(des.x, idx_var, brush)),
                idx_required
            )
        )
        t_solid = t_solid.at[
            from_list_id_couple_to_2tuples_ids(idx_resolving)
        ].set(TouchState.RESOLVING)

    # Step 8 - Find free SOLID touches
    idx_v_pos_and_ex = list(set().union(
        compress(idx_full_array, (p_void == PixelState.POSSIBLE).flatten()),
        compress(idx_full_array, (p_void == PixelState.EXISTING).flatten())
    ))
    idx_free = list(
        set(idx_full_array).difference(
            *map(
                lambda idx_var: set(get_convolved_idx(des.x, idx_var, brush)),
                idx_v_pos_and_ex
            )
        ).intersection(set(idx_valid))
    )
    # If at[[]] -> fulfil all the table
    if len(idx_free) > 0:
        t_solid = t_solid.at[
            from_list_id_couple_to_2tuples_ids(idx_free)].set(TouchState.FREE)

    # Bonus
    import matplotlib.pyplot as plt
    from scipy.signal import convolve2d
    fig, ax = plt.subplots(2, 3)
    ax[0, 0].imshow((p_solid == PixelState.POSSIBLE), cmap='Greys')
    ax[0, 1].imshow((p_void == PixelState.IMPOSSIBLE), cmap='Greys')
    ax[0, 2].imshow((p_solid == PixelState.POSSIBLE) & (p_void == PixelState.IMPOSSIBLE), cmap='Greys')

    ax[1, 0].imshow((p_void == PixelState.POSSIBLE), cmap='Greys')
    ax[1, 1].imshow((p_void == PixelState.EXISTING), cmap='Greys')
    ax[1, 2].imshow(convolve2d((p_void == PixelState.POSSIBLE)*1 + (p_void == PixelState.EXISTING)*1, brush, mode='same'), cmap='Greys')
    plt.show()

    return Design(des.reward, new_x, p_void, p_solid, t_void, t_solid)

# def add_void_touch(des: Design, idx_touch: int, brush):
#     idx_full_array = list(product(*map(range, des.x.shape)))
#     idx_dilated = get_convolved_idx(des.x, idx_touch, brush)

#     # No need to use the equation (B.5)
#     # --> By default, all touches are considered valid

#     # Step 1 - Update the new VOID touch
#     # des_updated = des.t_v.at[idx].set(TouchState.EXISTING)
#     # des.t_v[idx] = TouchState.FREE
#     t_void = des.t_v.at[idx_touch].set(TouchState.EXISTING)

#     # t_void = des.t_v.at[
#     #     from_list_2ids_to_2list_id(idx_touch)].set(TouchState.EXISTING)

#     # Step 2 - Update associated VOID pixels (B.3)
#     # p_v_existing = D(t_v, b)
#     # des.p_v[idx] = PixelState.EXISTING
#     p_void = des.p_v.at[
#         from_list_id_couple_to_2tuples_ids(idx_dilated)
#     ].set(PixelState.EXISTING)
#     # If pixels are already void, they cannot be solid
#     p_solid = des.p_s.at[
#         from_list_id_couple_to_2tuples_ids(idx_dilated)
#     ].set(PixelState.IMPOSSIBLE)

#     # Step 3 - Update SOLID touches
#     # --> Where a void touch exist, impossible to have a solid touch (B.4)
#     # t_s_impossible = D(p_v_existing, b)
#     idx_mapped = list(map(
#         lambda idx_var: get_convolved_idx(des.x, idx_var, brush),
#         idx_dilated
#     ))
#     idx_double_dilated = list(set().union(*idx_mapped))
#     t_solid = des.t_s.at[
#         from_list_id_couple_to_2tuples_ids(idx_double_dilated)
#     ].set(TouchState.INVALID)


#     # Step 4 - Then update SOLID pixels state
#     # --> Opposite of "possible solid pixels are those obtained by dilating
#     #                  all existing or valid touches" (B.6)
#     # p_s_possible = D(t_s | t_s_valid)
#     idx_t_s_ex_and_val = compress(
#         idx_full_array,
#         (t_solid == TouchState.VALID).flatten()
#     )
#     idx_t_s_ex_and_val_mapped = list(map(
#         lambda idx_var: get_convolved_idx(des.x, idx_var, brush),
#         idx_t_s_ex_and_val
#     ))
#     idx_double_dilated = list(set().union(*idx_t_s_ex_and_val_mapped))
#     idx_p_s_possible = idx_double_dilated
#     # p_solid = des.p_s.at[
#     #     from_list_id_couple_to_2tuples_ids(idx_double_dilated)
#     # ].set(PixelState.IMPOSSIBLE)



#     # return des_updated
#     return Design(des.reward, des.x, p_void, p_solid, t_void, t_solid)

# t_void = des.t_v.at[idx_touch].set(TouchState.EXISTING)

# # Step 2 - Update the new VOID pixel

# # p_void =
# # des_updated = des_updated.p_v.at[idx].set(PixelState.EXISTING)
# # des.p_v[idx] = PixelState.EXISTING
# p_void = des.p_v.at[idx_dilated].set(PixelState.EXISTING)

# idx_double_dilated = get_convolved_idx(des.x, idx_dilated, brush)
# print(idx_double_dilated)
# t_solid = des.t_s.at[idx_double_dilated].set(TouchState.INVALID)




    # # Step 4 - Then update VOID pixels state
    # # --> Opposite of "possible solid pixels are those obtained by dilating
    # #                  all existing or valid touches" (B.6)
    # # p_s_possible = D(t_s | t_s_valid)
    # idx_t__ex_and_val = compress(
    #     idx_full_array,
    #     (t_solid == TouchState.VALID).flatten()
    # )
    # idx_t_s_ex_and_val_mapped = list(map(
    #     lambda idx_var: get_convolved_idx(des.x, idx_var, brush),
    #     idx_t_s_ex_and_val
    # ))
    # idx_double_dilated = list(set().union(*idx_t_s_ex_and_val_mapped))
    # idx_p_s_possible = idx_double_dilated
    # # p_solid = des.p_s.at[
    # #     from_list_id_couple_to_2tuples_ids(idx_double_dilated)
    # # ].set(PixelState.IMPOSSIBLE)

    # PIXEL REQUIRED
    # TOUCH FREE
