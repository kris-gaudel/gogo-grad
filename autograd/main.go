package main

import (
	"fmt"
)

func main() {
	fmt.Println("Hello world!")

	// Testing the engine
	a := Init(2.0)
	b := Init(-3.0)
	c := Init(10.0)
	e := a.Mult(b)
	d := e.Add(c)
	f := Init(-2.0)
	L := d.Mult(f)

	L.Backward()
	fmt.Printf("dL/da = %f\n", a.grad)
	fmt.Printf("dL/db = %f\n", b.grad)
	fmt.Printf("dL/dc = %f\n", c.grad)
	fmt.Printf("dL/de = %f\n", e.grad)
	fmt.Printf("dL/dd = %f\n", d.grad)
	fmt.Printf("dL/df = %f\n", f.grad)
	fmt.Printf("dL/dL = %f\n", L.grad)

	// Testing the neuron
	// x := make([]*Value, 2)
	// x[0] = Init(2.0)
	// x[1] = Init(3.0)
	// n := InitNeuron(2, true)
	// y := n.Forward(x)
	// fmt.Println(y.data)

	// Testing the layer
	// x := make([]*Value, 2)
	// x[0] = Init(2.0)
	// x[1] = Init(3.0)
	// l := InitLayer(2, 3, true)
	// y := l.Forward(x)
	// for i := range y {
	// 	fmt.Println(y[i].data)
	// }

	// Testing the MLP
	x := make([]*Value, 3)
	x[0] = Init(2.0)
	x[1] = Init(3.0)
	x[2] = Init(-1.0)
	m := InitMLP(3, []int{4, 4, 1})
	y := m.Forward(x)
	fmt.Println(y[0].data)
}
