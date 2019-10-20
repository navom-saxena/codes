package dsAlgo;

import java.util.function.Consumer;
import java.util.function.Function;
import java.util.stream.Stream;

public class TestingFunctional {

    public static void main(String[] args) {

        GenericFunctionInterface<Integer, Boolean> isEven = n -> n % 2 == 0;
        System.out.println(isEven.func(5));

        GenericFunctionInterface<Integer, Integer> multiplicationTwo = n -> n * 2;
        System.out.println(multiplicationTwo.func(5));

        Stream.iterate(1, number -> number + 1)
                .map(number -> number * number)
                .limit(25)
                .forEach(number -> System.out.print(number + " "));

        Consumer<String> c = System.out::println;
        c.accept("hello");

        Function<Long, Long> adderLambda = (value) -> value + 3;
        Long resultLambda = adderLambda.apply((long) 8);
        System.out.println("resultLambda = " + resultLambda);

        TestInterface t0 = new TestInterface() {
            @Override
            public void testMethod() {
                System.out.println("in test");
            }
        };
        TestInterface t = () -> System.out.println("in test");
        TestInterface t2 = new TestInterface() {
            @Override
            public void testMethod() {
                System.out.println("in test2");
            }
        };

        System.out.println(t.getClass());
        System.out.println(t2.getClass());
    }

}
