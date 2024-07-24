// This script implements a set of functions to compute the receptive field of latent training patches to perform the prototype push operation //
// Adapted from https://github.com/cfchen-duke/ProtoPNet

function computeLayerRfInfo(
    layerFilterSize: number,
    layerStride: number,
    layerPadding: 'same' | 'valid' | number,
    previousLayerRfInfo: [number, number, number, number]
): [number, number, number, number] {
    const [nIn, jIn, rIn, startIn] = previousLayerRfInfo;
    const epsilon = 1e-7;

    let nOut, pad;
    if (layerPadding === 'same') {
        nOut = Math.ceil(nIn / (epsilon + layerStride));
        if (nIn % layerStride === 0) {
            pad = Math.max(layerFilterSize - layerStride, 0);
        } else {
            pad = Math.max(layerFilterSize - (nIn % (epsilon + layerStride)), 0);
        }
        console.assert(nOut === Math.floor((nIn - layerFilterSize + pad) / (epsilon + layerStride)) + 1);
        console.assert(pad === (nOut - 1)*layerStride - nIn + layerFilterSize);
    } else if (layerPadding === 'valid') {
        nOut = Math.ceil((nIn - layerFilterSize + 1) / (epsilon + layerStride));
        pad = 0;
    } else {
        pad = (layerPadding as number) * 2;
        nOut = Math.floor((nIn - layerFilterSize + pad) / (epsilon + layerStride)) + 1;
    }

    const pL = Math.floor(pad / 2);

    const jOut = jIn * layerStride;
    const rOut = rIn + (layerFilterSize - 1) * jIn;
    const startOut = startIn + ((layerFilterSize - 1) / 2 - pL) * jIn;

    return [nOut, jOut, rOut, startOut];
}

function computeRfProtoLAtSpatialLocation (
    imgSize: number,
    heightIndex: number,
    widthIndex: number, 
    protoLRfInfo: number[]
): [number, number, number, number] {
    const n = protoLRfInfo[0];
    const j = protoLRfInfo[1];
    const r = protoLRfInfo[2];
    const start = protoLRfInfo[3];
    console.assert(heightIndex < n);
    console.assert(widthIndex < n);

    const centerH = start + (heightIndex * j);
    const centerW =  start + (widthIndex * j);

    const rfStartHIndex = Math.max(Math.round(centerH - (r/2)), 0);
    const rfEndHIndex = Math.min(Math.round(centerH + (r/2)), imgSize);

    const rfStartWIndex = Math.max(Math.round(centerW - (r/2)), 0);
    const rfEndWIndex = Math.min(Math.round(centerW + (r/2)), imgSize);

    return [rfStartHIndex, rfEndHIndex, rfStartWIndex, rfEndWIndex]
};

export function computeRfPrototype (
    imgSize: number,
    prototypePatchIndex: number[],
    protoLRfInfo: number []
): [number, number, number, number, number] {
    const imgIndex = prototypePatchIndex[0];
    const heightIndex = prototypePatchIndex[1];
    const widthIndex = prototypePatchIndex[2];
    const rfIndices = computeRfProtoLAtSpatialLocation(imgSize, heightIndex, widthIndex, protoLRfInfo);

    return [imgIndex, rfIndices[0], rfIndices[1], rfIndices[2], rfIndices[3]]
};

export function computeProtoLayerRfInfo(
    img_size: number,
    layerFilterSizes: number[][],
    layerStrides: number[][],
    layerPaddings: ('same' | 'valid' | number)[],
    prototypeKernelSize: number
): number[] {
    console.assert(layerFilterSizes.length === layerStrides.length);
    console.assert(layerFilterSizes.length === layerPaddings.length);

    let rfInfo: [number, number, number, number] = [img_size, 1, 1, 0.5];
    //console.log('Filter size:', layerFilterSizes);
    for (let i = 0; i < layerFilterSizes.length; i++) {
        const filterSize = layerFilterSizes[i][0];
        const strideSize = layerStrides[i][0];
        const paddingSize = layerPaddings[i];

        rfInfo = computeLayerRfInfo(
            filterSize,
            strideSize,
            paddingSize,
            rfInfo
        );
    }
    //console.log('rfInfo previous:', rfInfo);
    const protoLayerRfInfo = computeLayerRfInfo(
        prototypeKernelSize,
        1,
        'valid',
        rfInfo
    );
    //console.log('protoLayerRfInfo:', protoLayerRfInfo);

    return protoLayerRfInfo;
};
