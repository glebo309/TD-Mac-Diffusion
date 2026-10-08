"""Install a GPU post-mask with alpha replacement and correct premultiplied edges.
Executed inside TouchDesigner; standard TOPs only, no GLSL or CPU frame processing.
"""
def install_input_mask(c):
    page=next(p for p in c.customPages if p.name=='Diffusion')
    if not hasattr(c.par,'Masktoinput'):
        p=page.appendToggle('Masktoinput',label='Mask to Input')[0]
        p.default=False;p.val=False
    if not hasattr(c.par,'Masksource'):
        p=page.appendMenu('Masksource',label='Mask Source')[0]
        p.menuNames=['alpha','brightness'];p.menuLabels=['Alpha','Brightness']
        p.default='brightness';p.val='brightness'
    def node(typ,name):return c.op(name) or c.create(typ,name)
    alpha=node(reorderTOP,'mask_mixed_alpha')
    alpha.inputConnectors[0].connect(c.op('mixed'))
    for channel in ['red','green','blue','alpha']:setattr(alpha.par,'output'+channel+'chan','alpha')
    alpha.par.format='rgba32float'
    denominator=node(levelTOP,'mask_safe_alpha')
    denominator.inputConnectors[0].connect(alpha)
    denominator.par.format='rgba32float';denominator.par.clampinput='unclamped'
    denominator.par.clamp=True;denominator.par.clamplow2=.000001;denominator.par.clamphigh2=1
    straight=node(compositeTOP,'mask_straight')
    straight.inputConnectors[0].connect(c.op('mixed'));straight.inputConnectors[1].connect(denominator)
    straight.par.operand='divide';straight.par.format='rgba32float'
    replace=node(reorderTOP,'mask_replace_alpha')
    replace.inputConnectors[0].connect(straight);replace.inputConnectors[1].connect(c.op('input_image'))
    replace.par.outputalpha='input2'
    replace.par.outputalphachan.expr="'alpha' if parent().par.Masksource == 'alpha' else 'luminance'"
    replace.par.format='rgba32float'
    premult=node(levelTOP,'mask_premultiply')
    premult.inputConnectors[0].connect(replace)
    premult.par.format='rgba32float';premult.par.clampinput='unclamped';premult.par.premultrgbbyalpha=True
    final=node(switchTOP,'mask_output')
    final.inputConnectors[0].connect(c.op('mixed'));final.inputConnectors[1].connect(premult)
    final.par.index.expr='int(parent().par.Masktoinput)'
    c.op('out1').inputConnectors[0].connect(final)
    if c.op('output_preview') is not None:c.op('output_preview').par.top=c.op('out1')
    for i,n in enumerate([alpha,denominator,straight,replace,premult,final]):
        n.nodeX=400+i*170;n.nodeY=-180
    return c
