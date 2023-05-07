from itertools import compress, product

from gruyere.design import Design
from gruyere.ops import get_convolved_idx
from gruyere.misc import from_list_id_couple_to_2tuples_ids
from gruyere.states import TouchState, PixelState, DesignState


def add_solid_touches(des: Design, idx_touches: list[int, int], brush):
    # Se restreindre aux pixels affectes
    # --> double dilatation de la touch
    # aka dilation de la zone de touch
    idx_mapped = list(map(
        lambda idx_var: get_convolved_idx(des.x, idx_var, brush),
        idx_t_dilated
    ))
    idx_t_double_dilated = list(set().union(*idx_mapped))


    # Prepare the new design
    new_x = des.x[idx_t_double_dilated]

    t_solid = des.t_s[idx_t_double_dilated]
    p_solid = des.p_s[idx_t_double_dilated]

    t_void = des.t_v[idx_t_double_dilated]
    p_void = des.p_v[idx_t_double_dilated]

    idx_t_dilated = list(set().union(
        *map(
            lambda idx_var: get_convolved_idx(des.x, idx_var, brush),
            idx_touches
        )
    ))
    couples_idx_t_dilated = from_list_id_couple_to_2tuples_ids(idx_t_dilated)

    # No need to use the equation (B.5)
    # --> By default, all touches are considered valid

    # Step 0 - Update the new SOLID touch
    t_solid[from_list_id_couple_to_2tuples_ids(idx_touches)] = TouchState.EXISTING

    # Step 1 - Update the design
    new_x[couples_idx_t_dilated] = DesignState.SOLID

    # Step 2 - Update associated SOLID pixels (B.3)
    # p_s_existing = D(t_s, b)
    p_solid[couples_idx_t_dilated] = PixelState.EXISTING
    # If pixels are already solid, they cannot be void
    # TODO: Check if it is necessary and correct
    # p_void = des.p_v.at[
    #     from_list_id_couple_to_2tuples_ids(idx_t_dilated)
    # ].set(PixelState.IMPOSSIBLE)

    # Step 3 - Update VOID touches
    # --> Where a solid touch exist, impossible to have a void touch (B.4)
    # t_v_impossible = D(p_s_existing, b)
    idx_mapped = list(map(
        lambda idx_var: get_convolved_idx(des.x, idx_var, brush),
        idx_t_dilated
    ))
    # idx_t_double_dilated = list(set().union(*idx_mapped))
    # # idx_full_array = idx_t_double_dilated
    t_void[from_list_id_couple_to_2tuples_ids(idx_t_double_dilated)] = TouchState.INVALID

    # Step 4 - Find valid SOLID touches
    idx_valid = list(compress(
        idx_full_array,
        # (
        #     (t_solid == TouchState.VALID) | ~(t_solid == TouchState.EXISTING)
        # ).flatten()
        ~(
            (t_solid == TouchState.INVALID) | (t_solid == TouchState.EXISTING)
        ).flatten()
    ))

    # If at[[]] -> fulfil all the table
    if len(idx_valid) > 0:
        # t_solid = t_solid.at[
        #     from_list_id_couple_to_2tuples_ids(idx_valid)
        # ].set(TouchState.VALID)
        t_solid[from_list_id_couple_to_2tuples_ids(idx_valid)] = TouchState.VALID

    # Step 5 - Find impossible VOID pixels
    idx_p_void_possible = list(set().union(
        *map(
            lambda idx_var: set(get_convolved_idx(des.x, idx_var, brush)),
            list(set().union(
                compress(idx_full_array, (t_void == TouchState.EXISTING).flatten()),
                compress(idx_full_array, (t_void == TouchState.VALID).flatten()),
            ))
        )
    ))
    idx_p_void_impossible = list(
        set(idx_full_array).difference(set(idx_p_void_possible))
    )
    if len(idx_p_void_impossible) > 0:
        # p_void = p_void.at[
        #     from_list_id_couple_to_2tuples_ids(idx_p_void_impossible)
        # ].set(PixelState.IMPOSSIBLE)
        p_void[from_list_id_couple_to_2tuples_ids(idx_p_void_impossible)] = PixelState.IMPOSSIBLE

    # Step 6 - Find required SOLID pixels
    idx_required = list(compress(
        idx_full_array,
        # ~(
        #     (p_solid == PixelState.EXISTING) | (p_void == PixelState.POSSIBLE)
        # ).flatten()
        (
            ((p_solid == PixelState.POSSIBLE)
             | (p_solid == PixelState.REQUIRED))
            & (p_void == PixelState.IMPOSSIBLE)
        ).flatten()
    ))
    # idx_required = list(set(idx_required).difference())

    # If at[[]] -> fulfil all the table
    if len(idx_required) > 0:
        p_solid[from_list_id_couple_to_2tuples_ids(idx_required)] = PixelState.REQUIRED
        p_void[from_list_id_couple_to_2tuples_ids(idx_required)] = PixelState.IMPOSSIBLE

    # Step 7 - Find resolving SOLID touches
    # If at[[]] -> fulfil all the table
    if len(idx_required) > 0:
        idx_resolving = set(idx_valid).intersection(
            *map(
                lambda idx_var: set(get_convolved_idx(des.x, idx_var, brush)),
                idx_required
            )
        )
        if len(idx_resolving) > 0:
            t_solid[from_list_id_couple_to_2tuples_ids(idx_resolving)] = TouchState.RESOLVING


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
        t_solid[from_list_id_couple_to_2tuples_ids(idx_free)] = TouchState.FREE

    return Design(des.reward, new_x, p_void, p_solid, t_void, t_solid)


def add_void_touches(des: Design, idxs: list[int, int], brush):
    des = add_solid_touches(des._invert(), idxs, brush)

    return des._invert()
