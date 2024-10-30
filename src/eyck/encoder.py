import latenta as la

def transfer_amortization_input(amortization_input, transcriptome, loader = None, verbose = True):
    if loader is None:
        loader = transcriptome.create_loader()

    transcriptome_definition = transcriptome.create_definition()

    print(f"""
    coords equal: {transcriptome_definition["gene"].coords == amortization_input["gene"].coords}
    length new  : {len(transcriptome_definition["gene"].coords)}
    length amort: {len(amortization_input["gene"].coords)}
    overlap     : {len(set(transcriptome_definition["gene"].coords) & set(amortization_input["gene"].coords))}
    """)

    coords_equal = transcriptome_definition["gene"].coords == amortization_input["gene"].coords

    amortization_input_raw = la.Fixed(loader, definition = transcriptome_definition, label = "amortization_input", transforms = [la.transforms.Log1p()], invert_value = False)

    if not coords_equal:
        amortization_input = la.operations.SelectAndImpute(x = amortization_input_raw, filtering = dict(gene = amortization_input.coords))
    else:
        amortization_input = amortization_input_raw

    return amortization_input

def transfer_encoder(latent, transcriptome, loader = None, verbose = True):
    latent = latent.clone()
    if loader is None:
        loader = transcriptome.create_loader()

    transcriptome_definition = transcriptome.create_definition()

    print(f"""
    coords equal: {transcriptome_definition["gene"].coords == latent.find("amortization_input")["gene"].coords}
    length new  : {len(transcriptome_definition["gene"].coords)}
    length amort: {len(latent.find("amortization_input")["gene"].coords)}
    overlap     : {len(set(transcriptome_definition["gene"].coords) & set(latent.find("amortization_input")["gene"].coords))}
    """)

    coords_equal = transcriptome_definition["gene"].coords == latent.find("amortization_input")["gene"].coords

    amortization_input_raw = la.Fixed(loader, definition = transcriptome_definition, label = "amortization_input_raw", transforms = [la.transforms.Log1p()], invert_value = False)

    if not coords_equal:
        amortization_input = la.operations.SelectAndImpute(x = amortization_input_raw, filtering = dict(gene = latent.find("amortization_input")["gene"].coords))
    else:
        amortization_input = amortization_input_raw

    latent.replace(latent.find("amortization_input"), amortization_input)

    loc = la.Fixed(latent.q.loc.prior_pd())
    scale = la.Fixed(latent.q.scale.prior_pd())

    latent = la.variables.Latent(
        q = la.distributions.Normal(loc, scale, transforms = latent.q.transforms),
        p = latent.p,
        label = latent.label,
        symbol = latent.symbol,
    )
    return latent