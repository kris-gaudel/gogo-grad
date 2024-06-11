package main

import (
	"fmt"
)

func main() {
	fmt.Println("Hello world!")

	a := InitValue(2.0)
	b := InitValue(-3.0)
	c := InitValue(10.0)
	d := a.Mult(b)
	d = d.Add(c)

	fmt.Println(d.Data)
}
