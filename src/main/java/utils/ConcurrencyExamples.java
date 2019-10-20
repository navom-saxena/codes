package utils;


import java.util.Arrays;
import java.util.List;
import java.util.concurrent.*;

public class ConcurrencyExamples {

    public static void main(String[] args) throws InterruptedException {

        for (int i = 0; i < 5; i++) {
            String h = "hello";
            CompletableFuture.runAsync(() -> {
                System.out.println(h + " -- " + Thread.currentThread().getName());
                throw new RuntimeException();
            }).exceptionally(throwable -> {
                System.out.println("here");
                throwable.printStackTrace();
                throw new RuntimeException(throwable);
            }).thenAccept((t) -> {
                System.out.println("in accept");
            });
        }

        Thread.sleep(10000);

        Runnable task = () -> {
            String threadName = Thread.currentThread().getName();
            System.out.println("Hello " + threadName);
        };

        task.run();
        Thread t1 = new Thread(task);
        t1.start();

        System.out.println("Done!");

        Runnable runnable = () -> {
            String threadName = Thread.currentThread().getName();
            System.out.println("Foo " + threadName);
            try {
                TimeUnit.SECONDS.sleep(1);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
            System.out.println("bar " + threadName);
        };

        Thread t2 = new Thread(runnable);
        t2.start();

        Callable<Integer> callable = () -> {
            try {
                TimeUnit.SECONDS.sleep(1);
                return 123;
            } catch (InterruptedException e) {
                e.printStackTrace();
                return 0;
            }
        };

        ExecutorService executorService = Executors.newSingleThreadExecutor();
        executorService.submit(() -> {
            String threadName = Thread.currentThread().getName();
            System.out.println("hello, triggered via executor service " + threadName);
        });

        Future<Integer> integerFuture = executorService.submit(callable);
        System.out.println("is integer future done " + integerFuture.isDone());

        try {
            Integer result = integerFuture.get();
            System.out.println("future result " + result);
        } catch (InterruptedException | ExecutionException e) {
            e.printStackTrace();
        }

        List<Callable<String>> callableList = Arrays.asList(
                () -> "task1",
                () -> "task2",
                () -> "task3");

        try {
            executorService.invokeAll(callableList).stream()
                    .map(stringFuture -> {
                        try {
                            return stringFuture.get();
                        } catch (InterruptedException | ExecutionException e) {
                            throw new RuntimeException(e);
                        }
                    }).forEach(System.out::println);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        ScheduledExecutorService scheduledExecutorService = Executors.newScheduledThreadPool(1);

        Runnable runnable1 = () -> System.out.println("Scheduling: " + System.nanoTime());
        ScheduledFuture<?> future = scheduledExecutorService.schedule(runnable1, 3, TimeUnit.SECONDS);

        try {
            TimeUnit.MILLISECONDS.sleep(13);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }

        long remainingDelay = future.getDelay(TimeUnit.MILLISECONDS);
        System.out.printf("Remaining Delay: %sms", remainingDelay);

        int initialDelay = 0;
        int period = 1;
        scheduledExecutorService.scheduleAtFixedRate(runnable1, initialDelay, period, TimeUnit.SECONDS);
        scheduledExecutorService.scheduleWithFixedDelay(runnable1, initialDelay, period, TimeUnit.SECONDS);

        scheduledExecutorService.shutdown();

        System.out.println("attempt to shutdown executor");
        executorService.shutdown();
        try {
            executorService.awaitTermination(5, TimeUnit.SECONDS);
        } catch (InterruptedException e) {
            e.printStackTrace();
        } finally {
            if (!executorService.isTerminated()) {
                System.err.println("cancel non finished tasks");
            }
            executorService.shutdownNow();
            System.out.println("shutdown finished");
        }
    }

}
