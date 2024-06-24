package main

type MLP struct {
	layers []*Layer
}

func InitMLP(nint int, nouts []int) *MLP {
	sz := append([]int{nint}, nouts...)
	layers := make([]*Layer, len(nouts)-1)
	for i := range layers {
		nonlin := i != len(nouts)-1
		layers[i] = InitLayer(sz[i], sz[i+1], nonlin)
	}
	return &MLP{layers: layers}
}

func (m *MLP) Forward(x []*Value) []*Value {
	for _, l := range m.layers {
		x = l.Forward(x)
	}
	return x
}

func (m *MLP) Parameters() []*Value {
	var params []*Value
	for _, l := range m.layers {
		params = append(params, l.Parameters()...)
	}
	return params
}

func (m *MLP) ZeroGrad() {
	for _, param := range m.Parameters() {
		param.grad = 0.0
	}
}
